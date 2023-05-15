import random

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Resize

from omegaconf import OmegaConf

from .lvcnet import LVCBlock


MAX_WAV_VALUE = 32768.0


def resize(tensor, w, h):
    _, W, H = tensor.shape
    return Resize(size=(w or W, h or H), antialias=True)(tensor[:, None])[:, 0]


class AE(nn.Module):

    def __init__(self, io, hidden=768, latent=32, downsample=4):
        super().__init__()
        self.downsample = downsample
        self.encoder = nn.Sequential(
            nn.Conv1d(io, hidden, 1),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden, latent, 1),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent, hidden, 1),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden, io, 1),
        )

    @torch.jit.unused
    def forward(self, x):
        xs = x.shape
        x = x.detach()
        z = self.encoder(x)
        if self.downsample > 1:
            _, _, T = z.shape
            z = resize(z, None, T // 4)
            z = resize(z, None, T)
        rec = self.decoder(z)
        assert xs == rec.shape
        loss = F.l1_loss(rec, x)
        return rec, loss


class Generator(nn.Module):
    """UnivNet Generator"""
    def __init__(self, hp):
        super(Generator, self).__init__()
        self.mel_channel = hp.audio.n_mel_channels
        self.noise_dim = hp.gen.noise_dim
        self.hop_length = hp.audio.hop_length
        channel_size = hp.gen.channel_size
        kpnet_conv_size = hp.gen.kpnet_conv_size
        kpnet_hidden_channels = hp.gen.kpnet_hidden_channels

        self.mel_ae_reg = hp.gen.mel_ae_reg
        self.mel_rand_lerp_reg = hp.gen.mel_rand_lerp_reg
        self.mel_noise_reg = hp.gen.mel_noise_reg

        # autoencoder regularization (simulates TTS mels)
        self.mel_ae = None
        if self.mel_ae_reg:
            self.mel_ae = AE(self.mel_channel)

        self.res_stack = nn.ModuleList()
        hop_length = 1
        for stride in hp.gen.strides:
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    hp.audio.n_mel_channels,
                    stride=stride,
                    dilations=hp.gen.dilations,
                    lReLU_slope=hp.gen.lReLU_slope,
                    cond_hop_length=hop_length,
                    kpnet_conv_size=kpnet_conv_size,
                    kpnet_hidden_channels=kpnet_hidden_channels,
                )
            )

        self.conv_pre = \
            nn.utils.weight_norm(nn.Conv1d(hp.gen.noise_dim, channel_size, 7, padding=3, padding_mode='reflect'))

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(hp.gen.lReLU_slope),
            nn.utils.weight_norm(nn.Conv1d(channel_size, 1, 7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )

    def forward(self, c, z):
        '''
        Args:
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length)
            z (Tensor): the noise sequence (batch, noise_dim, in_length)

        '''
        z = self.conv_pre(z)                # (B, c_g, L)

        # mel augmentations
        aux_loss = 0
        if not torch.jit.is_scripting() and self.training:

            # autoencoder regularization (simulates TTS mels)
            if self.mel_ae_reg:
                rec, ae_loss = self.mel_ae(c)
                c = random.choice([c, rec, (rec + c) / 2]).detach()
                aux_loss = aux_loss + ae_loss

            # random stretching (robustness to different inference mel framerates)
            if self.mel_rand_lerp_reg and random.random() > 0.8:
                _, _, T = c.shape
                r = random.choice([0.8, 0.9, 0.95])
                c = resize(c, None, int(T * r))
                c = resize(c, None, T)

            # random noise
            if self.mel_noise_reg and random.random() > 0.8:
                # more noise at higher freqs
                _, ch, _ = c.shape
                arange = torch.arange(ch, 0, -1, dtype=torch.float, device=c.device)
                noise_weight = 5 + 15 * (arange[None, :, None] / ch) ** 2
                c = c + torch.randn_like(c) / noise_weight

        for res_block in self.res_stack:
            # res_block.to(z.device)
            z = res_block(z, c)             # (B, c_g, L * s_0 * ... * s_i)

        z = self.conv_post(z)               # (B, 1, L * 256)

        return z, aux_loss

    @torch.jit.unused
    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    @torch.jit.unused
    def remove_weight_norm(self):
        print('Removing weight norm...')

        nn.utils.remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    @torch.jit.export
    def inference(self, c):
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129, device=c.device)
        mel = torch.cat((c, zero), dim=2)
        z = torch.randn(1, self.noise_dim, mel.size(2), device=mel.device)

        audio, _ = self.forward(mel, z)
        audio = audio.squeeze() # collapse all dimension except time axis
        audio = audio[:-(self.hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()

        return audio


if __name__ == '__main__':
    hp = OmegaConf.load('config/reg_univ_32kHz_c64.yaml')
    model = Generator(hp)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"num params: {pytorch_total_params:,}")

    c = torch.randn(3, 256, 20)
    z = torch.randn(3, 64, 20)

    print("forward test")
    y, _ = model(c, z)
    print(c.shape, '->', y.shape)

    print("inference test")
    model.eval(inference=True)
    c0 = c[:1]
    y0 = model.inference(c0)
    print(c0.shape, '->', y0.shape)

    print("TorchScript test")
    sm = torch.jit.script(model)
    y0 = sm.inference(c0)
    print(c0.shape, '->', y0.shape)
