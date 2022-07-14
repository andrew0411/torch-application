from network.resnet import *

def get_model(args, shape):
    model = eval(args.model)(
        shape,
        num_classes = args.num_classes,
        checkpoint_dir = args.checkpoint_dir,
        checkpoint_name = args.checkpoint_name,
        pretrained = args.pretrained,
        pretrained_path = args.pretrained_path
    )
    return model