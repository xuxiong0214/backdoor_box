import sys
sys.path.append("/home/Xuxiong/experiment")
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from models.resnet import ResNet18
import random
import numpy as np
from tools.utils import save_image
from load_data.cifar import cifar
from tools.progress import progress_bar
from models import vgg, densenet
from models import vitsmall as vit
import argparse
import os

def gen_pidx(targets, pratio, tlabel):
    pidx = np.zeros(len(targets))
    for (i, t) in enumerate(targets):
        if random.random() < pratio and t != tlabel:
            pidx[i] = 1
    return pidx

class Addtrigger(object):
    def __init__(self, alpha, trigger) -> None:
        self.alpha = alpha
        self.trigger = trigger
        
    def __call__(self, img):
        return self.addtrigger(img)
        
    def addtrigger(self, nor_img):
        bad_img = (1 - self.alpha) * nor_img + self.alpha * self.trigger
        return bad_img

class BLENDCIFAR(cifar.CIFAR):
    def __init__(self, pidx, tlabel, path, train, train_type=None, tf=None) -> None:
        super().__init__(path, train, train_type, tf)
        self.pidx = pidx
        self.tlabel = tlabel
        self.totensor = transforms.Compose([transforms.ToTensor()])
        # trigger_path = "../../attack/blended/triggers/hellokitty_32.png"
        # trigger = Image.open(trigger_path).convert("RGB")
        # trigger = self.totensor(trigger)
        trigger = rand_noise
        self.bd_transform = transforms.Compose([Addtrigger(alpha=0.1, trigger=trigger)])
        self.trigger = trigger
        
    
    def __getitem__(self, index):
        img, label = self.datas[index], self.labels[index]
        img = Image.fromarray(img)
        poisoned = 0
        target_label = label
        img = self.totensor(img)
        if self.pidx[index] == 1:
            poisoned = 1
            img = self.bd_transform(img)
            # target_label = self.tlabel
            target_label = (label + 1) % 10
        img = transforms.ToPILImage()(img)
        if not self.tf is None:
            img = self.tf(img)

        return img, poisoned, label, target_label
    

def train(model, bad_trainloader, device, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, _, _, targets) in enumerate(bad_trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(bad_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(opt, model, cln_testloader, bad_testloader, device, criterion, save_trigger, tlabel):
    model.eval()
    test_cln_loss = 0
    test_bad_loss = 0
    cln_correct = 0
    bad_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, ((cln_inputs, _, _, cln_targets), (bad_inputs, _, _, bad_targets)) in enumerate(zip(cln_testloader, bad_testloader)):
            cln_inputs, cln_targets = cln_inputs.to(device), cln_targets.to(device)
            bad_inputs, bad_targets = bad_inputs.to(device), bad_targets.to(device)
            cln_outputs = model(cln_inputs)
            bad_outputs = model(bad_inputs)
            cln_loss = criterion(cln_outputs, cln_targets)
            bad_loss = criterion(bad_outputs, bad_targets)

            total += cln_inputs.shape[0]

            test_cln_loss += cln_loss.item()
            _, cln_predicted = cln_outputs.max(1)
            cln_correct += cln_predicted.eq(cln_targets).sum().item()

            test_bad_loss += bad_loss.item()
            _, bad_predicted = bad_outputs.max(1)
            bad_correct += bad_predicted.eq(bad_targets).sum().item()

            progress_bar(batch_idx, len(cln_testloader), 'ClnLoss:%.3f | BadLoss: %.3f | ACC: %.3f%% | ASR: %.3f%%'
                         % (test_cln_loss/(batch_idx+1), test_bad_loss/(batch_idx+1), 100.*cln_correct/total, 100.*bad_correct/total))
        
        state_dict = {
            "netC": model.state_dict(),
            "ACC": 100.*cln_correct/total,
            "ASR": 100.*bad_correct/total,
            "alpha":0.1,
            "trigger":save_trigger
        }
        # save_path = f"../../checkpoints/cifar-blend-{opt.model}-target{tlabel}"
        save_path = f"../../checkpoints/cifar-blend-{opt.model}-targetall"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(state_dict, os.path.join(save_path, "state_dict.pt.tar"))


def main(opt):
    poison_ratio = 0.05
    tlabel = opt.t
    device = "cuda:3"

    train_tf = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


    cln_trainset = cifar.CIFAR(path="../../datasets/cifar10", train=True, train_type=1, tf=train_tf)
    train_pidx = gen_pidx(cln_trainset.labels, pratio=poison_ratio, tlabel=opt.t)
    bad_trainset = BLENDCIFAR(pidx=train_pidx, tlabel=tlabel, path="../../datasets/cifar10", train=True, train_type=1, tf=train_tf)


    save_trigger = bad_trainset.trigger

    cln_testset = cifar.CIFAR(path="../../datasets/cifar10", train=False, tf=test_tf)
    test_pidx = np.ones(len(cln_testset.datas))
    bad_testset = BLENDCIFAR(pidx=test_pidx, tlabel=tlabel, path="../../datasets/cifar10", train=False, tf=test_tf)

    bad_trainloader = DataLoader(bad_trainset, 256, shuffle=True, num_workers=6)
    cln_testloader = DataLoader(cln_testset, 256, shuffle=False, num_workers=6)
    bad_testloader = DataLoader(bad_testset, 256, shuffle=False, num_workers=6)

    # model = vgg.VGG("VGG16").to(device)
    if opt.model == "vgg16":
        print("Load Model -- VGG16")
        print(f"Load Target -- {opt.t}")
        model = vgg.VGG("VGG16").to(device)
        epoch = 100
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif opt.model == "resnet18":
        print("Load Model -- ResNet18")
        print(f"Load Target -- {opt.t}")
        model = ResNet18().to(device)
        epoch = 300
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif opt.model == "vit":
        print("Load Model -- ViT")
        print(f"Load Target -- {opt.t}")
        model = vit.ViT(
                        image_size = 32,
                        patch_size = 4,
                        num_classes = 10,
                        dim = int(512),
                        depth = 6,
                        heads = 8,
                        mlp_dim = 512,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        ).to(device)
        epoch = 300
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    elif opt.model == "densenet":
        print("Load Model -- DenseNet")
        print(f"Load Target -- {opt.t}")
        model = densenet.DenseNet121().to(device)
        epoch = 100
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = torch.nn.CrossEntropyLoss()
    

    for e in range(epoch):
        print('\nEpoch: %d' % e)
        print(" Train")
        train(model=model, bad_trainloader=bad_trainloader, device=device, optimizer=optimizer, criterion=criterion)
        if (e+1) % 50 == 0:
            print(" Test")
            test(opt=opt, model=model, cln_testloader=cln_testloader, bad_testloader=bad_testloader, device=device, criterion=criterion, save_trigger=save_trigger, tlabel=tlabel)


rand_noise = torch.randn([3, 32, 32])
parser = argparse.ArgumentParser()
parser.add_argument("--t", type=int, default=0)
opt = parser.parse_args()
# models = ["densenet", "resnet18", "vit", "vgg"]
opt.model = "resnet18"
main(opt=opt)
        

    

