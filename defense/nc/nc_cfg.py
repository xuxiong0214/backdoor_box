import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    #load box...
    parser.add_argument("--dataset", type=str, default="cifar")
    parser.add_argument("--tlabel", type=int, default=0)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--attack", type=str, default="badnets")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    return parser
