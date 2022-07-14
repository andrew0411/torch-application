import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # model architecture
    parser.add_argument('--model', default='ResNet18', help='model to use -> | ResNet18 | ResNet50 | ResNet101 |')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--checkpoint_name', type=str, default='ResNet18')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--pretrained_path', type=str, default=None)

    # data load
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='./datasets/')

    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=50)

    # optimizer
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--swa_start', type=int, default=50)
    parser.add_argument('--swa_lr', type=float, default=0.05)

    args = parser.parse_args()

    return args