import argparse


def get_argument():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument("--dataset", type=str, default="cifar")
    parser.add_argument("--tlabel", type=int, default=0)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--attack", type=str, default="badnets")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--attack_mode", type=str, default="all2one")
    
    parser.add_argument("--data_fraction", type=float, default=1.0)
    
    parser.add_argument("--hand_set_model_path", type=str, default=None)
    parser.add_argument("--set_arch", type=str, default=None)
    parser.add_argument("--internal_index", type=int, default=None)
    
    parser.add_argument("--ae_atk_succ_t", type=float, default=0.9)
    
    parser.add_argument("--ae_filter_num", type=int, default=32)
    parser.add_argument("--ae_num_blocks", type=int, default=4)
    
    parser.add_argument("--mask_size", type=float, default=0.03)
    parser.add_argument("--override_epoch", type=int, default=None)
    parser.add_argument("--ignore_dist", action='store_true')
    parser.add_argument("--p_loss_bound", type=float, default=0.15)
    parser.add_argument("--loss_std_bound", type=float, default=1)
    parser.add_argument("--asr_test_type", type=str, default="filter")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--EPSILON", type=float, default=1e-7)
    parser.add_argument("--use_norm", type=int, default=1)

    parser.add_argument("--mixed_value_threshold", type=float, default=-0.75)

    return parser
