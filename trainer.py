import argparse
import torch
from omegaconf import OmegaConf
from utils.train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    args = parser.parse_args()

    hp = OmegaConf.load(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    assert hp.audio.hop_length == 256, \
        'hp.audio.hop_length must be equal to 256, got %d' % hp.audio.hop_length

    torch.manual_seed(hp.train.seed)
    torch.cuda.manual_seed(hp.train.seed)
    train(args, args.checkpoint_path, hp, hp_str)
