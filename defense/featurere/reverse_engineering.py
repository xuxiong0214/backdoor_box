import torch
from torch import Tensor, nn
import torchvision
import os
import numpy as np
import torch.nn.functional as F

from models import unet_model
import random
import pilgram
from PIL import Image
from functools import reduce


class RegressionModel(nn.Module):
    def __init__(self, box, opt, init_mask):
        self._EPSILON = opt.EPSILON
        super(RegressionModel, self).__init__()

        if init_mask is not None:
            self.mask_tanh = nn.Parameter(torch.tensor(init_mask))

        _, _, classifier = box.get_state_dict()
        for param in classifier.parameters():
            param.requires_grad = False
        self.classifier = classifier

        self.example_features = None
        self.AE = unet_model.UNet(n_channels=3,num_classes=3,base_filter_num=opt.ae_filter_num, num_blocks=opt.ae_num_blocks)
        self.device = box.device
        self.AE.train()
        self.example_ori_img = None
        self.example_ae_img = None
        self.opt = opt

    def forward_ori(self, x, opt):
        features = self.classifier.from_input_to_features(x)
        out = self.classifier.from_features_to_output(features)
        return out, features

    def forward_flip_mask(self, x,opt):

        strategy = "flip"
        features = self.classifier.from_input_to_features(x)
        if strategy == "flip":
            features = (1 - opt.flip_mask) * features - opt.flip_mask * features
        elif strategy == "zero":
            features = (1 - opt.flip_mask) * features

        out = self.classifier.from_features_to_output(features)
        return out, features

    def forward_ae(self, x, opt):

        self.example_ori_img = x
        x_before_ae = x
        x = self.AE(x)
        x_after_ae = x
        self.example_ae_img = x

        features = self.classifier.from_input_to_features(x)
        out = self.classifier.from_features_to_output(features)

        self.example_features = features

        return out, features, x_before_ae, x_after_ae


    def forward_ae_mask_p(self, x, opt):
        mask = self.get_raw_mask(opt)
        self.example_ori_img = x
        x_before_ae = x
        x = self.AE(x)
        x_after_ae = x
        self.example_ae_img = x

        features = self.classifier.from_input_to_features(x)
        reference_features_index_list = np.random.choice(range(opt.all_features.shape[0]), features.shape[0], replace=True)
        reference_features = opt.all_features[reference_features_index_list]
        features_ori = features
        features = mask * features + (1-mask) * reference_features.reshape(features.shape)

        out = self.classifier.from_features_to_output(features)

        self.example_features = features_ori

        return out, features, x_before_ae, x_after_ae, features_ori

    def forward_ae_mask_p_test(self, x, opt):
        mask = self.get_raw_mask(opt)
        self.example_ori_img = x
        x_before_ae = x
        x = self.AE(x)
        x_after_ae = x
        self.example_ae_img = x

        features = self.classifier.from_input_to_features(x)
        bs = features.shape[0]
        index_1 = list(range(bs))
        random.shuffle(index_1)
        reference_features = features[index_1]
        features_ori = features
        features = mask * features + (1-mask) * reference_features.reshape(features.shape)
        out = self.classifier.from_features_to_output(features)
        self.example_features = features_ori

        return out, features, x_before_ae, x_after_ae, features_ori

    def get_raw_mask(self, opt):
        mask = nn.Tanh()(self.mask_tanh)
        bounded = mask / (2 + self._EPSILON) + 0.5
        return bounded

class Recorder:
    def __init__(self, opt):
        super().__init__()
        self.mixed_value_best = float("inf")
        self.mixed_value_list = []

def train(box, opt, init_mask):

    opt.weight_p = 1
    opt.weight_acc = 1
    opt.weight_std = 1
    opt.init_mask = init_mask

    recorder = Recorder(opt)
    regression_model = RegressionModel(box, opt, init_mask).to(opt.device)

    opt.epoch = 50
    if opt.override_epoch:
        opt.epoch = opt.override_epoch

    optimizerR = torch.optim.Adam(regression_model.AE.parameters(),lr=opt.lr,betas=(0.5,0.9))
    optimizerR_mask = torch.optim.Adam([regression_model.mask_tanh],lr=1e-1,betas=(0.5,0.9))

    regression_model.AE.train()
    recorder = Recorder(opt)
    process = train_step

    warm_up_epoch = 30
    for epoch in range(warm_up_epoch):
        process(regression_model, optimizerR, optimizerR_mask, recorder, epoch, opt, warm_up=True)

    for epoch in range(opt.epoch):
        process(regression_model, optimizerR, optimizerR_mask, recorder, epoch, opt)
        recorder.mixed_value_list.append(recorder.mixed_value_best.item())

    opt.trained_regression_model = regression_model
    torch.save(regression_model.AE.state_dict(), f"./defense/featurere/{box.dataset}/{box.attack}/featureRE_inversion.pt")

    return recorder, opt

def get_range(box, opt, init_mask):

    test_dataloader = opt.re_dataloader_total_fixed
    inversion_engine = RegressionModel(box, opt, init_mask).to(opt.device)

    features_list = []
    features_list_class = [[] for i in range(opt.num_classes)]
    for batch_idx, (inputs, _, _, labels) in enumerate(test_dataloader):
        inputs = inputs.to(opt.device)
        out, features = inversion_engine.forward_ori(inputs,opt)
        print(torch.argmax(out,dim=1))

        features_list.append(features)
        for i in range(inputs.shape[0]):
            features_list_class[labels[i].item()].append(features[i].unsqueeze(0))
    all_features = torch.cat(features_list,dim=0)
    opt.all_features = all_features
    print(all_features.shape)

    del features_list
    del test_dataloader

    weight_map_class = []
    for i in range(opt.num_classes):
        feature_mean_class = torch.cat(features_list_class[i],dim=0).mean(0)
        weight_map_class.append(feature_mean_class)

    opt.weight_map_class = weight_map_class
    del all_features
    del features_list_class

def train_step(regression_model, optimizerR, optimizerR_mask, recorder, epoch, opt, warm_up=False):
    print("Epoch {} - Label: {} | {} - {}:".format(epoch, opt.target_label, opt.dataset, opt.attack_mode))
    cross_entropy = nn.CrossEntropyLoss()
    total_pred = 0
    true_pred = 0

    loss_ce_list = []
    loss_dist_list = []
    loss_list = []
    acc_list = []

    p_loss_list = []
    loss_mask_norm_list = []
    loss_std_list = []

    for (inputs, _, _, _) in opt.re_dataloader_total_fixed:
        regression_model.AE.train()
        regression_model.mask_tanh.requires_grad = False

        optimizerR.zero_grad()

        inputs = inputs.to(opt.device)
        sample_num = inputs.shape[0]
        total_pred += sample_num
        target_labels = torch.ones((sample_num), dtype=torch.int64).to(opt.device) * opt.target_label
        if warm_up:
            predictions, features, x_before_ae, x_after_ae = regression_model.forward_ae(inputs,opt)
        else:
            predictions, features, x_before_ae, x_after_ae, features_ori = regression_model.forward_ae_mask_p(inputs,opt)

        loss_ce = cross_entropy(predictions, target_labels)

        mse_loss = torch.nn.MSELoss(size_average = True).to(opt.device)(x_after_ae,x_before_ae)

        if warm_up:
            dist_loss = torch.cosine_similarity(opt.weight_map_class[opt.target_label].reshape(-1),features.mean(0).reshape(-1),dim=0)
        else:
            dist_loss = torch.cosine_similarity(opt.weight_map_class[opt.target_label].reshape(-1),features_ori.mean(0).reshape(-1),dim=0)

        acc_list_ = []
        minibatch_accuracy_ = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() / sample_num
        acc_list_.append(minibatch_accuracy_)
        acc_list_ = torch.stack(acc_list_)
        avg_acc_G = torch.mean(acc_list_)

        acc_list.append(minibatch_accuracy_)

        p_loss = mse_loss
        p_loss_bound = opt.p_loss_bound
        loss_std_bound = opt.loss_std_bound

        atk_succ_threshold = opt.ae_atk_succ_t

        if opt.ignore_dist:
            dist_loss = dist_loss*0

        if warm_up:
            if (p_loss>p_loss_bound):
                total_loss = loss_ce + p_loss*100
            else:
                total_loss = loss_ce
        else:
            loss_std = (features_ori*regression_model.get_raw_mask(opt)).std(0).sum()
            loss_std = loss_std/(torch.norm(regression_model.get_raw_mask(opt), 1))

            total_loss = dist_loss*5
            if dist_loss<0:
                total_loss = total_loss - dist_loss*5
            if loss_std>loss_std_bound:
                total_loss = total_loss + loss_std*10*(1+opt.weight_std)
            if (p_loss>p_loss_bound):
                total_loss = total_loss + p_loss*10*(1+opt.weight_p)

            if avg_acc_G.item()<atk_succ_threshold:
                total_loss = total_loss + 1*loss_ce*(1+opt.weight_acc)

        total_loss.backward()
        optimizerR.step()

        mask_norm_bound = int(reduce(lambda x,y:x*y,opt.feature_shape)*opt.mask_size)

        if not warm_up:
            for k in range(1):
                regression_model.AE.eval()
                regression_model.mask_tanh.requires_grad = True

                optimizerR_mask.zero_grad()
                predictions, features, x_before_ae, x_after_ae, features_ori = regression_model.forward_ae_mask_p(inputs,opt)
                loss_mask_ce = cross_entropy(predictions, target_labels)
                loss_mask_norm = torch.norm(regression_model.get_raw_mask(opt), opt.use_norm)
                loss_mask_total = loss_mask_ce
                if loss_mask_norm>mask_norm_bound:
                    loss_mask_total = loss_mask_total + loss_mask_norm

                loss_mask_total.backward()
                optimizerR_mask.step()

        loss_ce_list.append(loss_ce.detach())
        loss_dist_list.append(dist_loss.detach())
        loss_list.append(total_loss.detach())

        true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()

        if not warm_up:
            p_loss_list.append(p_loss)
            loss_mask_norm_list.append(loss_mask_norm)
            loss_std_list.append(loss_std)

    loss_ce_list = torch.stack(loss_ce_list)
    loss_dist_list = torch.stack(loss_dist_list)
    loss_list = torch.stack(loss_list)
    acc_list = torch.stack(acc_list)

    avg_loss_ce = torch.mean(loss_ce_list)
    avg_loss_dist = torch.mean(loss_dist_list)
    avg_loss = torch.mean(loss_list)
    avg_acc = torch.mean(acc_list)

    if not warm_up:
        p_loss_list = torch.stack(p_loss_list)
        loss_mask_norm_list = torch.stack(loss_mask_norm_list)
        loss_std_list = torch.stack(loss_std_list)

        avg_p_loss = torch.mean(p_loss_list)
        avg_loss_mask_norm = torch.mean(loss_mask_norm_list)
        avg_loss_std = torch.mean(loss_std_list)
        print("avg_ce_loss:",avg_loss_ce)
        print("avg_asr:",avg_acc)
        print("avg_p_loss:",avg_p_loss)
        print("avg_loss_mask_norm:",avg_loss_mask_norm)
        print("avg_loss_std:",avg_loss_std)


        if avg_acc.item()<atk_succ_threshold:
            print("@avg_asr lower than bound")
        if avg_p_loss>1.0*p_loss_bound:
            print("@avg_p_loss larger than bound")
        if avg_loss_mask_norm>1.0*mask_norm_bound:
            print("@avg_loss_mask_norm larger than bound")
        if avg_loss_std>1.0*loss_std_bound:
            print("@avg_loss_std larger than bound")


        mixed_value = avg_loss_dist.detach() - avg_acc + max(avg_p_loss.detach()-p_loss_bound,0)/p_loss_bound + max(avg_loss_mask_norm.detach()-mask_norm_bound,0)/mask_norm_bound + max(avg_loss_std.detach()-loss_std_bound,0)/loss_std_bound
        print("mixed_value:",mixed_value)
        if mixed_value < recorder.mixed_value_best:
            recorder.mixed_value_best = mixed_value
        opt.weight_p = max(avg_p_loss.detach()-p_loss_bound,0)/p_loss_bound
        opt.weight_acc = max(atk_succ_threshold-avg_acc,0)/atk_succ_threshold
        opt.weight_std = max(avg_loss_std.detach()-loss_std_bound,0)/loss_std_bound


    print(
        "  Result: ASR: {:.3f} | Cross Entropy Loss: {:.6f} | Dist Loss: {:.6f} | Mixed_value best: {:.6f}".format(
            true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_dist, recorder.mixed_value_best
        )
    )

    recorder.final_asr = avg_acc
    
    return avg_acc

if __name__ == "__main__":
    pass
