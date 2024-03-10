import sys
sys.path.append("/home/Xuxiong/experiment")
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from models.resnet_imagenet import resnet18
import random
import numpy as np
from torchvision.utils import save_image
from load_data.imagenet import imagenet
from tools.progress import progress_bar
import os
import argparse

def gen_pidx(targets, pratio, tlabel):
    pidx = np.zeros(len(targets))
    for (i, t) in enumerate(targets):
        if random.random() < pratio and t != tlabel:
            pidx[i] = 1
    return pidx

class Addtrigger(object):
    def __init__(self, mask, ptn) -> None:
        self.mask = mask
        self.ptn = ptn
        
    def __call__(self, img):
        return self.addtrigger(img)
        
    def addtrigger(self, data):
        bad_img = (1 - self.mask) * data + self.mask * self.ptn
        return bad_img

class BADCIFAR(imagenet.ImageNet):
    def __init__(self, pidx, tlabel, path, train, tf=None) -> None:
        super().__init__(path, train, tf)
        self.pidx = pidx
        self.tlabel = tlabel
        self.bd_transform = transforms.Compose([transforms.Resize((256, 256)),
                                                transforms.RandomCrop(size=224, padding=4),
                                                transforms.ToTensor(),
                                                Addtrigger(mask, ptn),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                                    
    
    def __getitem__(self, index):
        img, label = self.datas[index], self.labels[index]
        img = Image.fromarray(img)
        poisoned = 0
        target_label = label
        if self.pidx[index] == 1:
            poisoned = 1
            img = self.bd_transform(img)
            target_label = self.tlabel
        else:
            img = self.tf(img)

        return img, poisoned, label, target_label
    
    

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
    scheduler.step()


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
            "mask": mask,
            "ptn": ptn
        }
        save_path = f"../../checkpoints/imagenet-badnets-resnet18-target{opt.t}"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(state_dict, os.path.join(save_path, "state_dict.pt.tar"))

parser = argparse.ArgumentParser()
parser.add_argument("--t", type=int, default=0)
opt = parser.parse_args()

poison_ratio = 0.05
tlabel = opt.t
batch_size = 128
device = "cuda:0"
ptn = torch.randn([3, 224, 224])
mask = torch.zeros(size=[1, 224, 224])
for i in range(15):
    for j in range(15):
        mask[0, 5+i, 5+j] = 1

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(size=224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

denormalizer = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])

cln_trainset = imagenet.ImageNet(path="../../datasets", train="poison", tf=train_tf)
train_pidx = gen_pidx(cln_trainset.labels, pratio=poison_ratio, tlabel=0)
bad_trainset = BADCIFAR(pidx=train_pidx, tlabel=tlabel, path="../../datasets/", train="poison", tf=train_tf)

cln_testset = imagenet.ImageNet(path="../../datasets", train="test", tf=test_tf)
test_pidx = np.ones(len(cln_testset.datas))
bad_testset = BADCIFAR(pidx=test_pidx, tlabel=tlabel, path="../../datasets/", train="test", tf=test_tf)

bad_trainloader = DataLoader(bad_trainset, 64, shuffle=True, num_workers=6)
print("bad_trainloader finished")
cln_testloader = DataLoader(cln_testset, 64, shuffle=False, num_workers=6)
print("cln_testloader finished")
bad_testloader = DataLoader(bad_testset, 64, shuffle=False, num_workers=6)
print("bad_testloader finished")

model = resnet18().to(device)
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100,200,300], 0.1)


for e in range(300):
    print('\nEpoch: %d' % e)
    print(" Train")
    train()
    if e % 50 == 0:
        print(" Test")
        test()
test()