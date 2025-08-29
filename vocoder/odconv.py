import torch
import torch.nn as nn

class ODConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, 4, 1),  # 4 heads
            nn.Sigmoid()
        )
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
    def forward(self, x):
        attn = self.attention(x)  # (B, 4, 1)
        x = self.conv(x)
        return x * attn.unsqueeze(-1) 