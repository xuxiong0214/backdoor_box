from reverse_engineering import *
from featurere_cfg import get_argument
import time
import pandas as pd
from loader import Box

def main():
    start_time = time.time()
    opt = get_argument().parse_args()
    box = Box(opt)

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 10
        opt.total_label = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        size = 32
        channel = 3
        opt.t_mean = torch.FloatTensor(mean).view(channel,1,1).expand(channel, size, size).cuda()
        opt.t_std = torch.FloatTensor(std).view(channel,1,1).expand(channel, size, size).cuda()
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
        opt.t_mean = torch.FloatTensor(mean).view(channel,1,1).expand(channel, size, size).cuda()
        opt.t_std = torch.FloatTensor(std).view(channel,1,1).expand(channel, size, size).cuda()
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
        opt.t_mean = torch.FloatTensor(mean).view(channel,1,1).expand(channel, size, size).cuda()
        opt.t_std = torch.FloatTensor(std).view(channel,1,1).expand(channel, size, size).cuda()

    else:
        raise Exception("Invalid Dataset")

    trainloader = box.get_dataloader(train="clean", batch_size=opt.batch_size, shuffle=True)
    opt.total_label = opt.num_classes

    dummy_model = RegressionModel(opt, None).to(opt.device)
    opt.feature_shape = []
    for batch_idx, (inputs, _, _, labels) in enumerate(trainloader):
        features = dummy_model.classifier.from_input_to_features(inputs.to(box.device))
        for i in range(1, len(features.shape)):
            opt.feature_shape.append(features.shape[i])
        break
    del dummy_model
    init_mask = torch.ones(opt.feature_shape)

    opt.pretrain_AE = None
    get_range(opt, init_mask)

    final_mixed_value_list = []
    mixed_value_list = {}

    for target in range(opt.num_classes):
        print("----------------- Analyzing all2one: target{}  -----------------".format(target))
        opt.target_label = target
        data_list = []
        for batch_idx, (inputs, _, _, labels) in enumerate(trainloader):
            print(batch_idx)
            print(inputs.shape)
            data_list.append(inputs)
        opt.data_now = data_list
        recorder, opt = train(opt, init_mask)
        final_mixed_value_list.append(recorder.mixed_value_best.item())
        mixed_value_list[str(target)] = recorder.mixed_value_list

    print("final_mixed_value_list:",final_mixed_value_list)

if __name__ == "__main__":
    main()
