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
from load_data.gtsrb import gtsrb
from tools.progress import progress_bar
# from models import vit, densenet
from models import vitsmall as vit
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--t", type=int, default=0)
opt = parser.parse_args()
rand_noise = torch.randn([3, 32, 32])
root = "/home/Xuxiong/experiment"

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

class Blend(gtsrb.GTSRB):
    def __init__(self, pidx, tlabel, path, train, train_type=None, tf=None) -> None:
        super().__init__(path, train, train_type, tf)
        self.pidx = pidx
        self.tlabel = tlabel
        trigger = rand_noise
        self.bd_transform = transforms.Compose([transforms.Resize((40, 40)),
                                                transforms.RandomCrop(size=32, padding=4),
                                                transforms.RandomHorizontalFlip(0.5),
                                                transforms.ToTensor(),
                                                Addtrigger(alpha=0.1, trigger=trigger),
                                                transforms.Normalize([0, 0, 0], [1, 1, 1])])
        
    def __getitem__(self, index):
        img = Image.open(self.datas[index])
        label = self.labels[index]
        poisoned = 0
        target_label = label
        if self.pidx[index] == 1:
            poisoned = 1
            img = self.bd_transform(img)
            target_label = self.tlabel
        else:
            img = self.tf(img)
            
        return img, poisoned, label, target_label
    
    
        
poison_ratio = 0.05
tlabel = opt.t
batch_size = 128
device = "cuda:0"

train_tf = transforms.Compose([transforms.Resize((40, 40)),
                               transforms.RandomCrop(size=32, padding=4),
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.ToTensor(),
                               transforms.Normalize([0, 0, 0], [1, 1, 1])])

test_tf = transforms.Compose([transforms.Resize((40, 40)),
                               transforms.CenterCrop(size=32),
                               transforms.ToTensor(),
                               transforms.Normalize([0, 0, 0], [1, 1, 1])])


cln_trainset = gtsrb.GTSRB(path=os.path.join(root, "datasets/gtsrb"), train=True, train_type=1, tf=train_tf)
train_pidx = gen_pidx(cln_trainset.labels, pratio=poison_ratio, tlabel=0)
bad_trainset = Blend(pidx=train_pidx, tlabel=tlabel, path=os.path.join(root, "datasets/gtsrb"), train=True, train_type=1, tf=train_tf)

cln_testset = gtsrb.GTSRB(path=os.path.join(root, "datasets/gtsrb"), train=False, tf=test_tf)
test_pidx = np.ones(len(cln_testset.datas))
bad_testset = Blend(pidx=test_pidx, tlabel=tlabel, path=os.path.join(root, "datasets/gtsrb"), train=False, tf=test_tf)

bad_trainloader = DataLoader(bad_trainset, 256, shuffle=True, num_workers=6)
cln_testloader = DataLoader(cln_testset, 256, shuffle=False, num_workers=6)
bad_testloader = DataLoader(bad_testset, 256, shuffle=False, num_workers=6)

# model = vgg.VGG("VGG16").to(device)
# model = ResNet18(num_classes=43).to(device)

model = vit.ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 43,
    dim = int(512),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)

# model = densenet.DenseNet121(num_classes=43)
# model.load_state_dict(torch.load("../../checkpoints/gtsrb-blend-vit-target0/state_dict.pt.tar", map_location="cpu")["netC"])
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
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


def test():
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
            "trigger":rand_noise
        }

        save_path = f"../../checkpoints/gtsrb-blend-vit-target{tlabel}"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(state_dict, os.path.join(save_path, "state_dict.pt.tar"))



for e in range(300):
    print('\nEpoch: %d' % e)
    print(" Train")
    train()
    if (e+1) % 50 == 0:
        print(" Test")
        test()
