import sys
sys.path.append("/home/Xuxiong/experiment")
from loader import Box
import argparse
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
from febmodels import CompletionNetwork
import os
from tools.progress import progress_bar
import torchvision as tv
# from networks import Generator
import numpy as np
import cv2
from evaluate import test
from torchvision.transforms import transforms
from utils import poisson_blend_old

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="gtsrb")
parser.add_argument("--tlabel", type=int, default=0)
parser.add_argument("--model", type=str, default="resnet18")
parser.add_argument("--attack", type=str, default="badnets")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--size", type=int, default=32) #width * height
parser.add_argument("--num_classes", type=int, default=43)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--attack_type", type=str, default="all-to-one")


opt = parser.parse_args()
box = Box(opt)
denormalizer = box.get_denormalizer()
params1, params2, net = box.get_state_dict()
net.to(box.device)
net.eval()
target_layers = [net.layer4[-1]]
cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
# cln_trainloader = box.get_dataloader(train="clean", batch_size=opt.batch_size, shuffle=True)
cln_testloader = box.get_dataloader(train="test", batch_size=opt.batch_size, shuffle=False)

config = {}
config["input_dim"] = 3
config["ngf"] = 32

netG = CompletionNetwork()
root = box.root
netG.load_state_dict(torch.load(os.path.join(root, "defense/februus/gtsrb_inpainting"), map_location="cpu"))
netG.to(box.device)
netG.eval()
total = 0
correct = 0
acc = 0
# mpv = [0.4914655575466156, 0.4821903321331739, 0.4465675537097454]
mpv = [0.33373367140503546, 0.3057189632961195, 0.316509230828686]
mpv = torch.tensor(mpv).view(1,3,1,1)
mpv = mpv.to(box.device)


# This funciton is GAN restoration module
def GAN_patching_inputs(images, box): # images and its predicted tensors
    global N
    model = CompletionNetwork()
    model.load_state_dict(torch.load("defense/februus/cifar10_inpainting", map_location='cpu'))
    model.eval()
    model = model.to(box.device)
    cleanimgs = list(range(len(images))) # GAN inpainted
    denormalizer = box.get_denormalizer()
    normalizer = box.get_normalizer()
    # This is to apply Grad CAM to the load images
    # --------------------------------------------
    for j in range(len(images)):
        # N += 1
        image = images[j]
        image = denormalizer(image) # unnormalize to [0 1] to feed into GAN
        image = torch.unsqueeze(image, 0) # unsqueeze meaning adding 1D to the tensor

        mask = cam(image)  # get the mask through GradCAM
        MASK_COND = 0.8 #cifar 0.7
        cond_mask = mask >= MASK_COND
        mask = cond_mask.astype(int)

        # ---------------------------------------
        # print(mask.shape)
        mask = np.expand_dims(mask,axis=0) # add 1D to mask
        # mask = np.expand_dims(mask,axis=0)
        mask = torch.tensor(mask) # convert mask to tensor 1,1,32,32
        mask = mask.type(torch.FloatTensor)
        mask = mask.to(box.device)
        x = image # original test image


        # mpv = [0.4914655575466156, 0.4821903321331739, 0.4465675537097454]
        mpv = [0.33373367140503546, 0.3057189632961195, 0.316509230828686]
        mpv = torch.tensor(mpv).view(1,3,1,1)
        mpv = mpv.to(box.device)
        # inpaint
        with torch.no_grad():
            x_mask = x - x * mask + mpv * mask # generate the occluded input [0 1]
            inputx = torch.cat((x_mask, mask), dim=1)
            output = model(inputx) # generate the output for the occluded input [0 1]

            # image restoration
            inpainted = poisson_blend_old(x_mask, output, mask) # this is GAN output [0 1]
            inpainted = inpainted.to(box.device)

            # store GAN output
            clean_input = inpainted
            clean_input = normalizer(clean_input) # normalize to [-1 1]
            clean_input = torch.squeeze(clean_input) # remove the 1st dimension
            cleanimgs[j] = clean_input.cpu().numpy() # store to a list

    # this is tensor for GAN output
    cleanimgs_tensor = torch.from_numpy(np.asarray(cleanimgs))
    cleanimgs_tensor = cleanimgs_tensor.type(torch.FloatTensor)
    cleanimgs_tensor = cleanimgs_tensor.to(box.device)

    return cleanimgs_tensor


for batch_idx, (cln_img, _, _, targets) in enumerate(cln_testloader):
    bsize = cln_img.shape[0]
    cln_img = cln_img.to(box.device)
    poi_img = box.poisoned(cln_img, param1=params1, param2=params2).to(box.device)
    # grayscale_cam = cam(input_tensor=poi_img, targets=None)
    # mask = torch.from_numpy(grayscale_cam).to(box.device)
    # mask = mask.reshape([bsize, 1, 32, 32])
    # x_mask = cln_img * (1-mask) + mpv*mask
    # inputx = torch.cat((x_mask, mask), dim=1)
    # pur_img = netG(inputx)
    pur_img = GAN_patching_inputs(poi_img, box)
    outputs = net(pur_img)

    
    # x1, x2, offset_flow = netG(poi_img, mask)
    # inpainted_result = x2 * mask + poi_img * (1. - mask)
    # inpainted_result = inpainted_result[:, :, :32, :32]

    _, predicted = outputs.max(1)
    for i in range(cln_img.shape[0]):
        total += 1
        p = predicted[i]
        t = targets[i]
        if t == box.tlabel:
            continue
        if p == t:
            correct += 1

    if total > 0:
        acc = 100.*correct/total
    else:
        acc = 0

    progress_bar(batch_idx, len(cln_testloader), 'Result: %.5f%%'
                    % (acc))
    
total = 0
correct = 0
acc = 0
for batch_idx, (cln_img, _, _, targets) in enumerate(cln_testloader):
    # bsize = cln_img.shape[0]
    # cln_img = cln_img.to(box.device)
    # poi_img = cln_img
    # grayscale_cam = cam(input_tensor=poi_img, targets=None)
    # mask = torch.from_numpy(grayscale_cam).to(box.device)
    # mask = mask.reshape([bsize, 1, 32, 32])
    # tv.utils.save_image(denormalizer(poi_img)[0], "ori_ia.png")
    # poi_img = poi_img*(1-mask)
    # mask = mask.repeat(1, 1, 7, 7)
    # poi_img = poi_img.repeat(1, 1, 7, 7)
    # mask = rz1(mask)
    # poi_img = rz1(poi_img)
    # tv.utils.save_image(denormalizer(poi_img)[0], "grad_ia.png")
    # x1, x2, offset_flow = netG(poi_img, mask)
    # inpainted_result = x2 * mask + poi_img * (1. - mask)
    # inpainted_result = inpainted_result[:, :, :32, :32]
    # inpainted_result = rz2(inpainted_result)
    # tv.utils.save_image(denormalizer(inpainted_result)[0], "pur_ia.png")
    # exit()
    cln_img = cln_img.to(box.device)
    pur_img = GAN_patching_inputs(cln_img, box)
    outputs = net(pur_img)
    _, predicted = outputs.max(1)
    for i in range(cln_img.shape[0]):
        total += 1
        p = predicted[i]
        t = targets[i]
        if p == t:
            correct += 1

    if total > 0:
        acc = 100.*correct/total
    else:
        acc = 0

    progress_bar(batch_idx, len(cln_testloader), 'Result: %.5f%%'
                    % (acc))