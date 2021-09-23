import argparse


def parse_args():

    parser = argparse.ArgumentParser(
        description="Tensorflow ImageNet Training")

    # General Hyperparameter Configuration

    parser.add_argument('-f')

    parser.add_argument('-- exp_name', default='test', type=str,
                        help='experiment_name')

    parser.add_argument('--train_epochs', type=int, default=1000,
                        help='Number of iteration')

    parser.add_argument('--train_steps', type=int, default=None,
                        help='Number base total steps iterate each epochs')

    parser.add_argument('--warmup_epochs', type=int, default=100,
                        help='Warmup the learning base period -- this Larger --> Warmup more slower')

    parser.add_argument('--dataset', metavar='DATA', default='tiny-imagenet',
                        choices=['cifar10', 'cifar100', 'tiny-imagenet', 'imagenet'])

    parser.add_argument('--learning_rate_scaling', metavar='learning_rate', default='no_scale',
                        choices=['linear', 'sqrt', 'no_scale', ])

    parser.add_argument('-j', '--num-workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')

    parser.add_argument('-- train_batch_size', default=512, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')

    return parser.parse_args()
