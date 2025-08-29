import torch
import torch.nn as nn
from .odconv import ODConv1d
from .grc_lora import GRC_LoRA_Block

class ModifiedHiFiGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.odconv1 = ODConv1d(80, 512, kernel_size=7, padding=3)
        self.mrf_blocks = nn.ModuleList([
            GRC_LoRA_Block(512, 512, kernel_size=3, dilation=1),
            GRC_LoRA_Block(512, 512, kernel_size=3, dilation=3),
            GRC_LoRA_Block(512, 512, kernel_size=3, dilation=5)
        ])
        self.conv_out = nn.Conv1d(512, 1, kernel_size=7, padding=3)
    def forward(self, mel):
        x = self.odconv1(mel)
        for block in self.mrf_blocks:
            x = block(x)
        return torch.tanh(self.conv_out(x)) 