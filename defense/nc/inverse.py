import sys
sys.path.append("/home/Xuxiong/experiment")
import torch
from torch import nn
from loader import Box
import cfg
from tools.progress import progress_bar
import time

class RegressionModel(nn.Module):
    def __init__(self, init_mask, init_pattern, box):
        self._EPSILON = 1e-7
        super(RegressionModel, self).__init__()
        self.mask_tanh = nn.Parameter(torch.tensor(init_mask))
        self.pattern_tanh = nn.Parameter(torch.tensor(init_pattern))
        self.box = box
        self.denormalizer = box.get_denormalizer()
        self.normalizer = box.get_normalizer()

    def forward(self, x):
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        if self.normalizer:
            pattern = self.normalizer(self.get_raw_pattern())
        x = (1 - mask) * x + mask * pattern
        return x

    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        return mask / (2 + self._EPSILON) + 0.5

    def get_raw_pattern(self):
        pattern = nn.Tanh()(self.pattern_tanh)
        return pattern / (2 + self._EPSILON) + 0.5

opt = cfg.get_arguments().parse_args()
device = opt.device
box = Box(opt)
save_path = box.get_save_path()
tlabel = box.tlabel

device = opt.device
box = Box(opt)
init_mask = torch.ones([1, box.size, box.size])
init_pattern = torch.ones([3, box.size, box.size])
regression_model = RegressionModel(init_mask, init_pattern, box)
optimizerR = torch.optim.Adam(regression_model.parameters(), lr=1e-2, betas=(0.5, 0.9))

cln_trainloader = box.get_dataloader(train="clean", batch_size=opt.batch_size, shuffle=True)
cln_testloader = box.get_dataloader(train="test", batch_size=opt.batch_size, shuffle=False)

regression_model.eval()
regression_model.to(box.device)

_, _, cls_model = box.get_state_dict()

cross_entropy = nn.CrossEntropyLoss()
mse = nn.MSELoss()
st = time.time()
best_reg = float("inf")
count = 0
cls_model.eval()

for n in range(100):
    print("Train:")
    total_loss = 0
    total_reg = 0
    for batch_idx, (cln_img, _, _, targets) in enumerate(cln_trainloader):
        optimizerR.zero_grad()
        cln_img = cln_img.to(box.device)
        targets = torch.ones_like(targets, device=box.device) * box.tlabel
        inv_img = regression_model(cln_img)
        outputs = cls_model(inv_img)
        loss_ce = cross_entropy(outputs, targets)
        loss_reg = torch.norm(regression_model.get_raw_mask(), 1)
        loss = loss_ce + 1e-3*loss_reg
        loss.backward()
        optimizerR.step()
        total_loss += loss.item()
        total_reg += loss_reg.item()
        progress_bar(batch_idx, len(cln_trainloader), 'loss:%.4f | mask_norm:%.4f' % (total_loss/(batch_idx+1), total_reg/(batch_idx+1)))

    
    if(total_reg < float("inf")):
        if(total_reg < 0.99*best_reg):
            best_reg = total_reg
            count += 1

    total = 0
    correct = 0
    print(count)

    best_mask = regression_model.get_raw_mask()
    best_pattern = regression_model.get_raw_pattern()
    torch.save(best_mask, f"./defense/topo/{box.dataset}-{box.attack}-mask.pt")
    torch.save(best_pattern, f"./defense/topo/{box.dataset}-{box.attack}-pattern.pt")
    # torch.save(regression_model.state_dict(), f"./defense/nc/{box.dataset}-{box.attack}-regression.pt")

    if count == 80:
        break


ed = time.time()
print(ed-st)