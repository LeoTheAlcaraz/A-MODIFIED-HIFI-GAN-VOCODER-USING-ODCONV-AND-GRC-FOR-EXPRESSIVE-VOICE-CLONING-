import torch
import torch.nn as nn

class GRC_LoRA_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, r=4):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=(kernel_size - 1) * dilation // 2, 
            dilation=dilation, groups=in_channels
        )
        self.lora_A = nn.Parameter(torch.randn(in_channels, r))
        self.lora_B = nn.Parameter(torch.randn(r, out_channels))
    def forward(self, x):
        base_output = self.conv(x)
        lora_adaptation = torch.matmul(self.lora_A, self.lora_B)
        return base_output + lora_adaptation.unsqueeze(0).unsqueeze(-1) * x + x 