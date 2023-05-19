import torch
import argparse
from pathlib import Path

from tqdm import tqdm
from omegaconf import OmegaConf

from datasets.dataloader import MelProcessor
from utils.utils import read_wav_np


def main(args):
    hp = OmegaConf.load(args.config)
    mel_proc = MelProcessor(hp)
    for wavpath in tqdm(args.input_folder.iterdir(), f"converting files"):
        try:
            sr, wav = read_wav_np(wavpath)
            mel = mel_proc(wav, sr)
            torch.save(mel, args.output_folder / wavpath.name)
        except KeyboardInterrupt as e:
            print("Exiting")
            return
        except Exception as e:
            print(f"Preproc encountered error on file {wavpath}, skipping. Error was: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, required=True,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-i', '--input_folder', type=Path, required=True,
                        help="directory of wavs / flacs to convert to mels.")
    parser.add_argument('-o', '--output_folder', type=Path, required=True,
                        help="directory which generated mels are saved.")
    args = parser.parse_args()
    main(args)
