import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='ResNet18', help='model to use -> | ResNet18 | ResNet50 | ResNet101 |')
    parser.add_argument('--num_classes', type=int, default=10)