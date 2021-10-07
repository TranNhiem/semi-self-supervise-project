import argparse


def parse_args():

    parser = argparse.ArgumentParser(
        description="Tensorflow ImageNet Training")

    # General Hyperparameter Configuration

    parser.add_argument('-f')

    parser.add_argument('-- exp_name', default='test', type=str,
                        help='experiment_name')

    parser.add_argument('--train_epochs', type=int, default=600,
                        help='Number of iteration')

    parser.add_argument('--classify_epochs', type=int, default=50,
                        help='Number of iteration')

    parser.add_argument('--train_steps', type=int, default=None,
                        help='Number base total steps iterate each epochs')

    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Warmup the learning base period -- this Larger --> Warmup more slower')

    parser.add_argument('--dataset', metavar='DATA', default='tiny-imagenet',
                        choices=['cifar10', 'cifar100', 'tiny-imagenet', 'imagenet'])

    parser.add_argument('--learning_rate_scaling', metavar='learning_rate', default='linear',
                        choices=['linear', 'sqrt', 'no_scale', ])

    parser.add_argument('-j', '--num-workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')

    parser.add_argument('--train_batch_size', default=50, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')

    # Optimizer Configs:
    # In optimizer we will have three Option ('Original Configure', 'Weight Decay', 'Gradient Centralization')
    parser.add_argument('--optimizer', type=str, default="LARSW_GC", help="Optimization for update the Gradient",
                        choices=['Adam', 'SGD', 'LARS', 'AdamW', 'SGDW', 'LARSW',
                                 'AdamGC', 'SGDGC', 'LARSGC', 'AdamW_GC', 'SGDW_GC', 'LARSW_GC'])
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="Momentum manage how fast of update Gradient")

    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help="weight_decay to penalize the update gradient")

    # Configure for Conv_Trasnformer Architecture

    # Configure for Self-Supervised Training

    parser.add_argument('--SSL_training', type=str, default="ssl_train", help="Optimization for update the Gradient",
                        choices=['ssl_train', 'classify_train'])

    return parser.parse_args()
