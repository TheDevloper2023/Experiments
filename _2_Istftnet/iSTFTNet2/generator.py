"""Generator and residual blocks for the iSTFTNet vocoder."""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d, ConvTranspose2d
from torch.nn.utils import weight_norm, remove_weight_norm

from utils import init_weights, get_padding

LRELU_SLOPE = 0.1


class ShuffleBlock2D(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        pass

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for conv in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)



########### Start ISTFTNET2 shit


# the paper is weirdly gatekeeping this module, so I asked Deepseek to write it for me.
# I am sorry
class ID2DBridge(nn.Module):
    """
    Bundles the 1D↔2D frequency plumbing for iSTFTNet2:
        to_2d: (B, concat_ch, T)     -> (B, c_2d, T, freq_low)
        to_1d: (B, c_2d, T, freq_low)-> (B, 2*(n_fft//2+1), T)
    Owns: conv_2d_in (1x1 projection), freq_up (transposed conv).
    """
    def __init__(self, concat_ch, n_fft, freq_low=8, c_2d=None):
        super().__init__()
        assert concat_ch % freq_low == 0, \
            f"concat_ch={concat_ch} not divisible by freq_low={freq_low}"

        self.freq_low   = freq_low
        self.n_fft      = n_fft
        self.freq_out   = n_fft // 2 + 1
        self.c_2d_in    = concat_ch // freq_low
        self.c_2d       = c_2d if c_2d is not None else self.c_2d_in

        # 1D -> 2D
        self.conv_2d_in = weight_norm(Conv2d(self.c_2d_in, self.c_2d, 1))
        self.conv_2d_in.apply(init_weights)

        # 2D -> (mag, phase) spectrum
        kernel_f = self.freq_out - (freq_low - 1) * freq_low
        assert kernel_f > 0, \
            f"freq_low={freq_low}, n_fft={n_fft} give kernel_f={kernel_f} (invalid)"
        self.freq_up = weight_norm(ConvTranspose2d(
            self.c_2d, 2, kernel_size=(1, kernel_f),
            stride=(1, freq_low), padding=(0, 0)))
        self.freq_up.apply(init_weights)

    def to_2d(self, x):
        B, F, T = x.shape
        x = x.view(B, self.c_2d_in, self.freq_low, T)     # (B, c_2d_in, freq_low, T)
        x = x.permute(0, 1, 3, 2).contiguous()            # (B, c_2d_in, T, freq_low)
        return self.conv_2d_in(x)                          # (B, c_2d, T, freq_low)

    def to_1d(self, x):
        x = self.freq_up(x)                                # (B, 2, T, freq_out)
        B, _, T, F = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B, 2 * F, T)     # (B, 2*freq_out, T)
        split = self.freq_out
        return torch.exp(x[:, :split, :]), torch.sin(x[:, split:, :])



# 2D RESNET and 2D ShuffleNet

class ResNet2D_Block(nn.Module):
    def __init__(self, channels, kernel_size=3, n_units=3):
        super().__init__()

        self.units = nn.ModuleList()

        for _ in range(n_units):
            self.units.append(
                nn.Sequential(
                    nn.LeakyReLU(LRELU_SLOPE),
                    weight_norm(nn.Conv2d(channels, channels * 2, kernel_size=kernel_size, padding=kernel_size // 2)),
                    nn.LeakyReLU(LRELU_SLOPE),
                    weight_norm(nn.Conv2d(channels * 2, channels, kernel_size=kernel_size, padding=kernel_size // 2)),
                )
            )

    def forward(self, x):
        for unit in self.units:
            x = x + unit(x)
        return x

    def remove_weight_norm(self):
        for unit in self.units:
            remove_weight_norm(unit[1])
            remove_weight_norm(unit[3])


########## End of ISTFTNET2 modulz

class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                kernel_size, upsample_rate,
                                padding=(kernel_size - upsample_rate) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            channels = h.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(resblock(h, channels, kernel_size, dilation))

        self.post_n_fft = h.gen_istft_n_fft
        self.ups.apply(init_weights)


        self.bridge_2d = ID2DBridge(
            concat_ch= channels * self.num_kernels,
            n_fft=self.post_n_fft,
            freq_low=getattr(h, "freq_low", 8),
            c_2d=getattr(h, "c_2d", None)
        ) 

        # 2D stuff yay
        block_2d_type = getattr(h, "block_2d", "res")
        
        if block_2d_type == "res":
            self.resnet2d = ResNet2D_Block(
                channels=self.bridge_2d.c_2d,
                kernel_size=3,
                n_units=3,
            )
        elif block_2d_type == "shuffle":
            raise NotImplementedError()
        else: # Shuffle Res Block 2d
            raise ValueError("No such 2D blocks as ", getattr(h, "block_2d"))

        remove_weight_norm(self.bridge_2d.conv_2d_in)
        remove_weight_norm(self.bridge_2d.freq_up)
        for block in [self.resnet2d]:
            block.remove_weight_norm()

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            branches = []
            for j in range(self.num_kernels):
                branches.append(self.resblocks[i * self.num_kernels + j](x))
            
            x = torch.cat(branches, dim=1)

        x = F.leaky_relu(x, LRELU_SLOPE)

        x = self.bridge_2d.to_2d(x)

        x = self.resnet2d(x)

        
        return self.bridge_2d.to_1d(x)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
