import sys
sys.path.append("/home/Xuxiong/experiment")
import torch
from loader import Box
from evaluate import test
from tools.progress import progress_bar
import sp_cfg
from torchvision import transforms
from copy import deepcopy


opt = sp_cfg.get_arguments().parse_args()
opt.attack_type = "all-to-one"
device = opt.device
box = Box(opt)
save_path = box.get_save_path()
_, _, classifier = box.get_state_dict()
tlabel = box.tlabel
shrink_size = box.size + 4
cln_testloader = box.get_dataloader(train="test", batch_size=opt.batch_size, shuffle=False)

def test(testloader, testmodel, box, poisoned = False, tlabel = None, passlabel=None):
    model = deepcopy(testmodel)
    model.eval()        
    correct = 0
    total = 0

    denormalizer = box.get_denormalizer()
    normalizer = box.get_normalizer()
    shrink = transforms.Resize((box.size + opt.shrink_size, box.size + opt.shrink_size))
    pad = transforms.RandomCrop(box.size, padding=2)

    if poisoned:
        param1, param2, _ = box.get_state_dict()
    with torch.no_grad():
        for batch_idx, (inputs, _, _, targets) in enumerate(testloader):
            inputs, targets = inputs.to(box.device), targets.to(box.device)
            ori_target = targets
            if poisoned:
                inputs = box.poisoned(inputs, param1, param2)

            inputs = denormalizer(inputs)
            inputs = shrink(inputs)
            inputs = pad(inputs)
            inputs = normalizer(inputs)

            if not tlabel is None:
                targets = torch.ones_like(targets, device=box.device) * tlabel

            outputs = model(inputs)

            _, predicted = outputs.max(1)

            for i in range(inputs.shape[0]):
                if (not passlabel is None) and ori_target[i] == passlabel:
                    continue
                total += 1
                p = predicted[i]
                t = targets[i]
                if p == t:
                    correct += 1

            if total > 0:
                acc = 100.*correct/total
            else:
                acc = 0

            progress_bar(batch_idx, len(testloader), 'Result: %.5f%%'
                         % (acc))
    
    return 100.*correct/total


ba = test(testloader=cln_testloader, testmodel=classifier, box=box, poisoned=False, passlabel=tlabel)
asr = test(testloader=cln_testloader, testmodel=classifier, box=box, poisoned=True, tlabel=tlabel, passlabel=tlabel)
