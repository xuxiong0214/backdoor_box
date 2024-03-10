import sys
sys.path.append("/home/Xuxiong/experiment")
from loader import Box
import argparse
from evaluate import test
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import torch
import torchvision as tv
from febmodels import CompletionNetwork, ContextDiscriminator
from torch.optim import Adadelta
from utils import gen_input_mask, gen_hole_area, crop
from losses import completion_network_loss
from tools.progress import progress_bar
from torch.nn import BCELoss, DataParallel


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar")
parser.add_argument("--tlabel", type=int, default=0)
parser.add_argument("--model", type=str, default="resnet18")
parser.add_argument("--attack", type=str, default="badnets")
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--size", type=int, default=32) #width * height
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=96)
parser.add_argument("--attack_type", type=str, default="all-to-one")
parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument('--gamma', type=float, default=1e-8)
parser.add_argument('--trigger-norm', type=float, default=100)
parser.add_argument('--steps_1', type=int, default=20)
parser.add_argument('--steps_2', type=int, default=20)
parser.add_argument('--steps_3', type=int, default=20)
# parser.add_argument('--snaperiod_1', type=int, default=10000)
# parser.add_argument('--snaperiod_2', type=int, default=2000)
# parser.add_argument('--snaperiod_3', type=int, default=10000)
parser.add_argument('--hole_min_w', type=int, default=2)
parser.add_argument('--hole_max_w', type=int, default=22)
parser.add_argument('--hole_min_h', type=int, default=2)
parser.add_argument('--hole_max_h', type=int, default=22)
parser.add_argument('--cn_input_size', type=int, default=32)
parser.add_argument('--ld_input_size', type=int, default=22) #cifar 0.7*32  gtsrb 0.8*32
parser.add_argument('--max_holes', type=int, default=1)


opt = parser.parse_args()
box = Box(opt)
alpha = 4e-4

_, _, net = box.get_state_dict()
net.to(box.device)
net.eval()
target_layers = [net.layer4[-1]]
cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)

cln_trainloader = box.get_dataloader(train="poison", batch_size=opt.batch_size, shuffle=True)
cln_testloader = box.get_dataloader(train="test", batch_size=opt.batch_size, shuffle=False)


mpv = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1,3,1,1).to(box.device)
alpha = torch.tensor(alpha, dtype=torch.float32).to(box.device)
model_cn = CompletionNetwork()
model_cn.to(box.device)
opt_cn = Adadelta(model_cn.parameters())
# opt_cn = torch.optim.Adam(model_cn.parameters(), lr=1e-3)

# train1
for step_1 in range(opt.steps_1):
    model_cn.train()
    for idx, (x, _, _, _) in enumerate(cln_trainloader):
        opt_cn.zero_grad()
        x = x.to(box.device)
        mask = gen_input_mask(
            shape=[x.shape[0], 1, x.shape[2], x.shape[3]],
            hole_size=(
                (opt.hole_min_w, opt.hole_max_w),
                (opt.hole_min_h, opt.hole_max_h)
            ),
            hole_area=None,
            max_holes=opt.max_holes
        ).to(box.device)
        x_mask = x*(1-mask) + mpv*mask
        inputs = torch.cat((x_mask, mask), dim=1)
        outputs = model_cn(inputs)
        loss = completion_network_loss(x, outputs, mask)
        loss.backward()
        opt_cn.step()
        progress_bar(idx, len(cln_trainloader), 'Phase 1 | Epoch %d | loss:%.4f'
                         % (step_1, loss.item()))

model_cd = ContextDiscriminator(
    local_input_shape=(3, opt.ld_input_size, opt.ld_input_size),
    global_input_shape=(3, opt.cn_input_size, opt.cn_input_size),
    arc=opt.dataset
)
model_cd.to(box.device)
opt_cd = Adadelta(model_cd.parameters())
# opt_cd = torch.optim.Adam(model_cd.parameters(), lr=1e-3)
bceloss = BCELoss()

for step_2 in range(opt.steps_2):
    model_cn.eval()
    model_cd.train()
    for idx, (x, _, _, _) in enumerate(cln_trainloader):
        x = x.to(box.device)
        hole_area_fake = gen_hole_area(
            (opt.ld_input_size, opt.ld_input_size),
            (x.shape[3], x.shape[2]))
        mask = gen_input_mask(
            shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
            hole_size=(
                (opt.hole_min_w, opt.hole_max_w),
                (opt.hole_min_h, opt.hole_max_h)),
            hole_area=hole_area_fake,
            max_holes=opt.max_holes).to(box.device)
        
        fake = torch.zeros((len(x), 1)).to(box.device)
        x_mask = x - x * mask + mpv * mask
        input_cn = torch.cat((x_mask, mask), dim=1)
        output_cn = model_cn(input_cn)
        input_gd_fake = output_cn.detach()
        input_ld_fake = crop(input_gd_fake, hole_area_fake)
        output_fake = model_cd((
                input_ld_fake.to(box.device),
                input_gd_fake.to(box.device)))
        loss_fake = bceloss(output_fake, fake)
        hole_area_real = gen_hole_area(
                (opt.ld_input_size, opt.ld_input_size),
                (x.shape[3], x.shape[2]))
        real = torch.ones((len(x), 1)).to(box.device)
        input_gd_real = x
        input_ld_real = crop(input_gd_real, hole_area_real)
        output_real = model_cd((input_ld_real, input_gd_real))
        loss_real = bceloss(output_real, real)
        # reduce
        loss = (loss_fake + loss_real) / 2.
        # backward
        loss.backward()

        progress_bar(idx, len(cln_trainloader), 'Phase 2 | Epoch %d | loss:%.4f'
                         % (step_2, loss.item()))

for step_3 in range(opt.steps_3):
    model_cn.eval()
    model_cd.train()
    for idx, (x, _, _, _) in enumerate(cln_trainloader):
        opt_cd.zero_grad()
        x = x.to(box.device)
        hole_area_fake = gen_hole_area(
            (opt.ld_input_size, opt.ld_input_size),
            (x.shape[3], x.shape[2]))
        mask = gen_input_mask(
            shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
            hole_size=(
                (opt.hole_min_w, opt.hole_max_w),
                (opt.hole_min_h, opt.hole_max_h)),
            hole_area=hole_area_fake,
            max_holes=opt.max_holes).to(box.device)

        # fake forward
        fake = torch.zeros((len(x), 1)).to(box.device)
        x_mask = x - x * mask + mpv * mask
        input_cn = torch.cat((x_mask, mask), dim=1)
        output_cn = model_cn(input_cn)
        input_gd_fake = output_cn.detach()
        input_ld_fake = crop(input_gd_fake, hole_area_fake)
        output_fake = model_cd((input_ld_fake, input_gd_fake))
        loss_cd_fake = bceloss(output_fake, fake)

        # real forward
        hole_area_real = gen_hole_area(
            (opt.ld_input_size, opt.ld_input_size),
            (x.shape[3], x.shape[2]))
        real = torch.ones((len(x), 1)).to(box.device)
        input_gd_real = x
        input_ld_real = crop(input_gd_real, hole_area_real)
        output_real = model_cd((input_ld_real, input_gd_real))
        loss_cd_real = bceloss(output_real, real)

        # reduce
        loss_cd = (loss_cd_fake + loss_cd_real) * alpha / 2.

        # backward model_cd
        loss_cd.backward()
        opt_cd.step()

        opt_cn.zero_grad()
        loss_cn_1 = completion_network_loss(x, output_cn, mask)
        input_gd_fake = output_cn
        input_ld_fake = crop(input_gd_fake, hole_area_fake)
        output_fake = model_cd((input_ld_fake, (input_gd_fake)))
        loss_cn_2 = bceloss(output_fake, real)

        # reduce
        loss_cn = (loss_cn_1 + alpha * loss_cn_2) / 2.

        # backward model_cn
        loss_cn.backward()
        opt_cn.step()
        progress_bar(idx, len(cln_trainloader), 'Phase 3 | Epoch %d | loss_cd:%.4f | loss_cn:%.4f'
                         % (step_3, loss_cd.item(), loss_cn.item()))
        
    torch.save(model_cn.state_dict(), f"februus_{opt.dataset}.pt")




# for batch_idx, (cln_img, _, _, targets) in enumerate(cln_testloader):
#     sz = cln_img.shape[0]
#     grayscale_cam = cam(input_tensor=cln_img, targets=None)
#     grayscale_cam = torch.from_numpy(grayscale_cam).reshape([sz, 1, 32, 32])
#     res = 0.5*denor(cln_img) + 0.5*grayscale_cam
#     # print(grayscale_cam.shape)
#     tv.utils.save_image(res[0:9], "gradcam.png", nrow=3)
#     # print(grayscale_cam.shape)
#     # print(cln_img.shape)
#     exit()
