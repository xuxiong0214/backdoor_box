import sys
sys.path.append("/home/Xuxiong/experiment")
from reverse_engineering import *
import time
from featurere_cfg import get_argument
from loader import Box
from tools.progress import progress_bar

def test(testloader, opt, box, poisoned = False, tlabel = None, passlabel=None, flip=False):
    trained_regression_model = opt.trained_regression_model
    trained_regression_model.eval()
    trained_regression_model.AE.eval()
    trained_regression_model.classifier.eval()
    flip_mask = trained_regression_model.get_raw_mask(opt)
    correct = 0
    total = 0
    if poisoned:
        param1, param2, _ = box.get_state_dict()
    with torch.no_grad():
        for batch_idx, (inputs, _, _, targets) in enumerate(testloader):
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            ori_target = targets
            if poisoned:
                inputs = box.poisoned(inputs, param1, param2)
            if not tlabel is None:
                targets = torch.ones_like(targets).to(opt.device) * tlabel

            if flip:
                features = trained_regression_model.classifier.from_input_to_features(inputs)
                features = (1 - flip_mask) * features - flip_mask * features
                outputs = trained_regression_model.classifier.from_features_to_output(features)
                _, predicted = outputs.max(1)
            else:
                features = trained_regression_model.classifier.from_input_to_features(inputs)
                outputs = trained_regression_model.classifier.from_features_to_output(features)
                _, predicted = outputs.max(1)

            for i in range(inputs.shape[0]):
                if (not passlabel is None) and (ori_target[i] == passlabel):
                    continue
                total += 1
                p = predicted[i]
                t = targets[i]
                if p == t:
                    correct += 1
            
            if total <= 0:
                total = 1

            progress_bar(batch_idx, len(testloader), 'Result: %.5f%%'
                         % (100.*correct/total))
            
    return 100.*correct/total

def main():
    start_time = time.time()
    opt = get_argument().parse_args()
    opt.attack_type = "all-to-one"

    # opt.tlabel = 0
    box = Box(opt)

    if opt.dataset == "cifar":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 10
        opt.total_label = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        size = 32
        channel = 3
        opt.t_mean = torch.FloatTensor(mean).view(channel,1,1).expand(channel, size, size).to(opt.device)
        opt.t_std = torch.FloatTensor(std).view(channel,1,1).expand(channel, size, size).to(opt.device)
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 43
        opt.total_label = 43
        mean = [0,0,0]
        std = [1,1,1]
        size = opt.input_height
        channel = 3
        opt.t_mean = torch.FloatTensor(mean).view(channel,1,1).expand(channel, size, size).to(opt.device)
        opt.t_std = torch.FloatTensor(std).view(channel,1,1).expand(channel, size, size).to(opt.device)
    elif opt.dataset == "imagenet":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
        opt.num_classes = 100
        opt.total_label = 100
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        size = 224
        channel = 3
        opt.t_mean = torch.FloatTensor(mean).view(channel,1,1).expand(channel, size, size).to(opt.device)
        opt.t_std = torch.FloatTensor(std).view(channel,1,1).expand(channel, size, size).to(opt.device)
    else:
        raise Exception("Invalid Dataset")

    trainloader = box.get_dataloader(train="clean", batch_size=opt.batch_size, shuffle=True)
    
    opt.total_label = opt.num_classes
    opt.re_dataloader_total_fixed = trainloader

    dummy_model = RegressionModel(box, opt, None).to(opt.device)
    opt.feature_shape = []
    for batch_idx, (inputs, _, _, _) in enumerate(trainloader):
        features = dummy_model.classifier.from_input_to_features(inputs.to(opt.device))
        for i in range(1, len(features.shape)):
            opt.feature_shape.append(features.shape[i])
        break
    del dummy_model
    init_mask = torch.ones(opt.feature_shape)

    opt.pretrain_AE = None
    get_range(box, opt, init_mask)

    final_mixed_value_list = []

    target = opt.tlabel
    print("----------------- Analyzing all2one: target{}  -----------------".format(target))
    opt.target_label = target
    data_list = []
    for batch_idx, (inputs, _, _, _) in enumerate(trainloader):
        data_list.append(inputs)
    opt.data_now = data_list
    recorder, opt = train(box, opt, init_mask)
    cln_testloader = box.get_dataloader(train="test", batch_size=opt.batch_size, shuffle=False)
    print("Test BA")
    ba = test(cln_testloader, opt, box, poisoned=False, flip=True)
    print("Test ASR")
    asr = test(cln_testloader, opt, box, poisoned=True, tlabel=opt.tlabel, flip=True, passlabel=0)

    final_mixed_value_list.append(recorder.mixed_value_best)

    return ba, asr

ba, asr = main()