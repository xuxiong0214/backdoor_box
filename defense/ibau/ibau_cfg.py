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
    
    ## hyper params
    parser.add_argument('--n_rounds', default=5, type=int, help='the maximum number of unelarning rounds')
    parser.add_argument('--K', default=5, type=int, help='the maximum number of fixed point iterations')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate of outer loop optimizer')


    return parser
