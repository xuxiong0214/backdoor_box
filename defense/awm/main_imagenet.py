import sys
sys.path.append("/home/Xuxiong/experiment")
import torch
from loader import Box
import argparse
import torch.nn.functional as F
from evaluate import test
import maskedconv
import os
from resnet_awm import ResNet18
from resnet_imagenet_awm import resnet18

def Regularization(model):
    L1=0
    L2=0
    for name, param in model.named_parameters():
        if 'mask' in name:
            L1 += torch.sum(torch.abs(param))
            L2 += torch.norm(param, 2)
    return L1, L2

def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)

def mask_train(model, criterion, mask_opt, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0

    batch_pert = torch.zeros([1,3,224,224], requires_grad=True, device=device)

    batch_opt = torch.optim.SGD(params=[batch_pert], lr=10)

    for i, (images, _, _, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        # step 1: calculate the adversarial perturbation for images
        ori_lab = torch.argmax(model.forward(images),axis = 1).long()
        per_logits = model.forward(images + batch_pert)
        loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
        loss_regu = torch.mean(-loss)

        batch_opt.zero_grad()
        loss_regu.backward(retain_graph = True)
        batch_opt.step()

    pert = batch_pert * min(1, opt.trigger_norm / torch.sum(torch.abs(batch_pert)))
    pert = pert.detach()

    for i, (images, _, _, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)
        
        perturbed_images = torch.clamp(images + pert[0], min=0, max=1)
        
        # step 2: calculate noisy loss and clean loss
        mask_opt.zero_grad()
        
        output_noise = model(perturbed_images)
        
        output_clean = model(images)
        pred = torch.argmax(output_clean, axis = 1).long()

        loss_rob = criterion(output_noise, labels)
        loss_nat = criterion(output_clean, labels)
        L1, L2 = Regularization(model)

        print("loss_noise | ", loss_rob.item(), " | loss_clean | ", loss_nat.item(), " | L1 | ", L1.item())
        loss = opt.alpha * loss_nat + (1 - opt.alpha) * loss_rob + opt.gamma * L1

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()

        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar")
parser.add_argument("--tlabel", type=int, default=0)
parser.add_argument("--model", type=str, default="resnet18")
parser.add_argument("--attack", type=str, default="badnets")
parser.add_argument("--device", type=str, default="cuda:3")
parser.add_argument("--size", type=int, default=32) #width * height
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--attack_type", type=str, default="all-to-one")
parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument('--gamma', type=float, default=1e-8)
parser.add_argument('--trigger-norm', type=float, default=100)

opt = parser.parse_args()
box = Box(opt)
device = box.device
classifier = resnet18()
root = "/home/Xuxiong/experiment"
folder = opt.dataset + "-" + opt.attack + "-" + opt.model + "-target" + str(opt.tlabel)
state_dict_path = os.path.join(root, "checkpoints/" + folder + "/state_dict.pt.tar")
state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
criterion = torch.nn.CrossEntropyLoss().to(device)

try:
    classifier.load_state_dict(state_dict["netC"], strict=False)
except:
    classifier.load_state_dict(state_dict["model"], strict=False)

classifier = classifier.to(device)
cln_trainloader = box.get_dataloader(train="clean", batch_size=128, shuffle=True)
cln_testloader = box.get_dataloader(train="test", batch_size=opt.batch_size, shuffle=False)

for name, module in classifier.named_modules():
        if isinstance(module, maskedconv.MaskedConv2d):
            module.include_mask()

parameters = list(classifier.named_parameters())

mask_params = [v for n, v in parameters if "mask" in n]
mask_names = [n for n, v in parameters if "mask" in n]
mask_optimizer = torch.optim.Adam(mask_params, lr = 1e-2)

for i in range(50):
    lr = mask_optimizer.param_groups[0]['lr']
    train_loss, train_acc = mask_train(model=classifier, criterion=criterion, data_loader=cln_trainloader, mask_opt=mask_optimizer)
    test(testloader=cln_testloader, testmodel=classifier, box=box, poisoned=False, poitarget=False)
    test(testloader=cln_testloader, testmodel=classifier, box=box, poisoned=True, poitarget=True)


# python defense/awm/main.py --dataset cifar --tlabel 0 --device cuda:0 --attack badnets --alpha 0.99
# python defense/awm/main.py --dataset cifar --tlabel 0 --device cuda:1 --attack wanet --alpha 0.99
# python defense/awm/main.py --dataset cifar --tlabel 0 --device cuda:2 --attack ia --alpha 0.99
# python defense/awm/main.py --dataset cifar --tlabel 0 --device cuda:0 --attack blend --alpha 0.99
# python defense/awm/main.py --dataset cifar --tlabel 0 --device cuda:3 --attack bppattack --alpha 0.99
# python defense/awm/main.py --dataset cifar --tlabel 0 --device cuda:3 --attack lc --alpha 0.99

# python defense/awm/main.py --dataset gtsrb --tlabel 0 --device cuda:1 --attack badnets --alpha 0.99 --num_classes 43
# python defense/awm/main.py --dataset gtsrb --tlabel 0 --device cuda:0 --attack blend --alpha 0.99 --num_classes 43
# python defense/awm/main.py --dataset gtsrb --tlabel 0 --device cuda:2 --attack ia --alpha 0.99 --num_classes 43
# python defense/awm/main.py --dataset gtsrb --tlabel 0 --device cuda:3 --attack wanet --alpha 0.99 --num_classes 43
# python defense/awm/main.py --dataset gtsrb --tlabel 0 --device cuda:0 --attack bppattack --alpha 0.99 --num_classes 43
# python defense/awm/main.py --dataset gtsrb --tlabel 0 --device cuda:2 --attack lc --alpha 0.99 --num_classes 43

# python defense/awm/main_imagenet.py --dataset imagenet --tlabel 0 --device cuda:1 --attack badnets --alpha 0.99 --num_classes 100
# python defense/awm/main_imagenet.py --dataset imagenet --tlabel 0 --device cuda:0 --attack blend --alpha 0.99 --num_classes 100
# python defense/awm/main_imagenet.py --dataset imagenet --tlabel 0 --device cuda:2 --attack ia --alpha 0.99 --num_classes 100
# python defense/awm/main_imagenet.py --dataset imagenet --tlabel 0 --device cuda:3 --attack wanet --alpha 0.99 --num_classes 100
# python defense/awm/main_imagenet.py --dataset imagenet --tlabel 0 --device cuda:0 --attack bppattack --alpha 0.99 --num_classes 100
# python defense/awm/main_imagenet.py --dataset imagenet --tlabel 0 --device cuda:2 --attack lc --alpha 0.99 --num_classes 100