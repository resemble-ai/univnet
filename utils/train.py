import os
import time
import logging
import tqdm
import torch
import itertools

from accelerate import Accelerator
from datasets.dataloader import create_dataloader
from utils.writer import MyWriter
from utils.stft import TacotronSTFT
from utils.stft_loss import MultiResolutionSTFTLoss
from model.generator import Generator
from model.discriminator import Discriminator

from .utils import get_commit_hash
from .validation import validate


def train(args, chkpt_path, hp, hp_str):
    accelerator = Accelerator()
    # args.num_gpus

    torch.cuda.manual_seed(hp.train.seed)

    model_g = Generator(hp)
    model_d = Discriminator(hp)

    optim_g = torch.optim.AdamW(model_g.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))
    optim_d = torch.optim.AdamW(model_d.parameters(),
        lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))

    githash = get_commit_hash()

    init_epoch = -1
    step = 0

    # define logger, writer, valloader, stft at rank_zero
    if accelerator.is_main_process:
        pt_dir = os.path.join(hp.log.chkpt_dir, args.name)
        log_dir = os.path.join(hp.log.log_dir, args.name)
        os.makedirs(pt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (args.name, time.time()))),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger()
        writer = MyWriter(hp, log_dir)
        valloader = create_dataloader(hp, args, train=False)
        val_stft = TacotronSTFT(
            filter_length=hp.audio.filter_length,
            hop_length=hp.audio.hop_length,
            win_length=hp.audio.win_length,
            n_mel_channels=hp.audio.n_mel_channels,
            sampling_rate=hp.audio.sampling_rate,
            mel_fmin=hp.audio.mel_fmin,
            mel_fmax=hp.audio.mel_fmax,
            center=False,
        )
        val_stft = accelerator.prepare(val_stft)

    if chkpt_path is not None:
        if accelerator.is_main_process:
            logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model_g.load_state_dict(checkpoint['model_g'])
        model_d.load_state_dict(checkpoint['model_d'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        # optim_d.load_state_dict(checkpoint['optim_d'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if accelerator.is_main_process:
            if hp_str != checkpoint['hp_str']:
                logger.warning("New hparams is different from checkpoint. Will use new.")

            if githash != checkpoint['githash']:
                logger.warning("Code might be different: git hash is different.")
                logger.warning("%s -> %s" % (checkpoint['githash'], githash))

    else:
        if accelerator.is_local_main_process:
            logger.info("Starting new training run.")

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    trainloader = create_dataloader(hp, args, train=True)

    model_g.train()
    model_d.train()

    resolutions = eval(hp.mrd.resolutions)
    stft_criterion = MultiResolutionSTFTLoss(resolutions)

    # accelerate wrappers
    model_g, model_d, optim_g, optim_d, trainloader, stft_criterion = accelerator.prepare(
        model_g, model_d, optim_g, optim_d, trainloader, stft_criterion)

    # save helper
    def save():
        save_path = os.path.join(pt_dir, '%s_%04d.pt' % (args.name, epoch))
        accelerator.save({
            'model_g': accelerator.unwrap_model(model_g).state_dict(),
            'model_d': accelerator.unwrap_model(model_d).state_dict(),
            'optim_g': optim_g.state_dict(),
            'optim_d': optim_d.state_dict(),
            'step': step,
            'epoch': epoch,
            'hp_str': hp_str,
            'githash': githash,
        }, save_path)
        logger.info("Saved checkpoint to: %s" % save_path)

    # init_step = step
    for epoch in itertools.count(init_epoch+1):
        trainloader.dataset.shuffle_mapping()

        # epoch val (NB: accelerate will fail to launch correctly if validation is done immediately!!)
        # if step > init_step + 10 and accelerator.is_main_process and epoch % hp.log.validation_interval == 0:
        if accelerator.is_main_process and epoch % hp.log.validation_interval == 0:
            validate(hp, model_g, model_d, valloader, val_stft, writer, step, accelerator.device)

        pbar = tqdm.tqdm(trainloader) if accelerator.is_local_main_process else trainloader
        for mel, audio in pbar:
            optim_g.zero_grad()

            # step val (NB: accelerate will fail to launch correctly if validation is done immediately!!)
            if accelerator.is_main_process and step % hp.log.validation_step_interval == 0:
                validate(hp, model_g, model_d, valloader, val_stft, writer, step, accelerator.device)

            # generator
            noise = torch.randn(hp.train.batch_size, hp.gen.noise_dim, mel.size(2))

            fake_audio, aux_loss = model_g(mel, noise)

            # Multi-Resolution STFT Loss
            sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
            stft_loss = (sc_loss + mag_loss) * hp.train.stft_lamb

            res_fake, period_fake = model_d(fake_audio)

            score_loss = 0.0
            for (_, score_fake) in res_fake + period_fake:
                score_loss = score_loss + torch.mean(torch.pow(score_fake - 1.0, 2))

            score_loss = score_loss / len(res_fake + period_fake)

            loss_g = score_loss + stft_loss + aux_loss

            # LR schedule
            if 500_000 < step <= 1_000_000:
                r = (step - 500_000) / 500_000
                lr = 1e-4 * r + 1e-5 * (1 - r)
                for g in optim_g.param_groups:
                    g['lr'] = lr

            accelerator.backward(loss_g)
            optim_g.step()

            # discriminator

            optim_d.zero_grad()
            res_fake, period_fake = model_d(fake_audio.detach())
            res_real, period_real = model_d(audio)

            loss_d = 0.0
            for (_, score_fake), (_, score_real) in zip(res_fake + period_fake, res_real + period_real):
                loss_d += torch.mean(torch.pow(score_real - 1.0, 2))
                loss_d += torch.mean(torch.pow(score_fake, 2))

            loss_d = loss_d / len(res_fake + period_fake)

            accelerator.backward(loss_d)
            optim_d.step()

            step += 1

            # logging
            if accelerator.is_local_main_process and step % hp.log.summary_interval == 0:
                loss_g = loss_g.item()
                loss_d = loss_d.item()
                writer.log_training(loss_g, loss_d, stft_loss.item(), score_loss.item(), step)
                trainloader.set_description("g %.04f d %.04f | step %d" % (loss_g, loss_d, step))

            # if step >= max_step:
            #     break

        if accelerator.is_main_process and epoch % hp.log.save_interval == 0:
            save()

        # if step >= max_step:
        #     break

    if accelerator.is_local_main_process:
        logger.info("Waiting for all processes...")
    accelerator.wait_for_everyone()
    save()
