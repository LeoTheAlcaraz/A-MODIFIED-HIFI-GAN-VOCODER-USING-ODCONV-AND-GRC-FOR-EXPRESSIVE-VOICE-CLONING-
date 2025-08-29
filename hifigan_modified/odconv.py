import torch
import torch.nn as nn
import torch.nn.functional as F

class ODConv1d(nn.Module):
    """
    Omni-Dimensional Dynamic Convolution for 1D speech data
    Implements dynamic kernel modulation across 4 dimensions:
    - Kernel number (K)
    - Spatial position (temporal)
    - Input channels (C_in)
    - Output channels (C_out)
    
    Based on Li et al. (2022) adapted for 1D speech processing
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, K=4, reduction_factor=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.K = K  # Number of parallel kernels
        
        # Multi-kernel convolution
        self.kernels = nn.Parameter(
            torch.randn(K, out_channels, in_channels, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(K, out_channels))
        
        # Attention mechanisms for 4 dimensions
        self.kernel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, K, 1),
            nn.Softmax(dim=1)
        )
        
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, kernel_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.in_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // reduction_factor, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_factor, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.out_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels // reduction_factor, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels // reduction_factor, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize convolution weights and attention parameters"""
        for kernel in self.kernels:
            nn.init.kaiming_normal_(kernel, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        """
        Forward pass with dynamic convolution
        Args:
            x: Input tensor (B, C_in, T)
        Returns:
            Output tensor (B, C_out, T_out)
        """
        B, C_in, T = x.shape
        
        # Simplified attention mechanism
        # Kernel attention: which kernel to emphasize
        kernel_attn = self.kernel_attention(x)  # (B, K, 1)
        
        # Apply dynamic convolution with attention
        outputs = []
        for k in range(self.K):
            # Get kernel weights for this kernel
            kernel_k = self.kernels[k]  # (C_out, C_in, kernel_size)
            bias_k = self.bias[k]  # (C_out,)
            
            # Simple convolution without complex attention
            output_k = F.conv1d(
                x, kernel_k, bias_k, 
                stride=self.stride, padding=self.padding, 
                dilation=self.dilation, groups=self.groups
            )
            
            # Weight by kernel attention
            output_k = output_k * kernel_attn[:, k:k+1, :]
            outputs.append(output_k)
        
        # Combine all kernel outputs
        output = sum(outputs)
        
        return output

class ODConvTranspose1d(nn.Module):
    """
    Omni-Dimensional Dynamic Transposed Convolution for upsampling
    Used in HiFi-GAN generator for temporal resolution increase
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 output_padding=0, dilation=1, groups=1, K=4, reduction_factor=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.K = K
        
        # Multi-kernel transposed convolution
        self.kernels = nn.Parameter(
            torch.randn(K, in_channels, out_channels, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(K, out_channels))
        
        # Attention mechanisms (same as ODConv1d)
        self.kernel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, K, 1),
            nn.Softmax(dim=1)
        )
        
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, kernel_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.in_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // reduction_factor, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_factor, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.out_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels // reduction_factor, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels // reduction_factor, out_channels, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize transposed convolution weights"""
        for kernel in self.kernels:
            nn.init.kaiming_normal_(kernel, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        """
        Forward pass with dynamic transposed convolution
        Args:
            x: Input tensor (B, C_in, T)
        Returns:
            Output tensor (B, C_out, T_out)
        """
        B, C_in, T = x.shape
        
        # Compute attention weights
        kernel_attn = self.kernel_attention(x)  # (B, K, 1)
        
        # Apply dynamic transposed convolution with attention
        outputs = []
        for k in range(self.K):
            kernel_k = self.kernels[k]  # (C_in, C_out, kernel_size)
            bias_k = self.bias[k]  # (C_out,)
            
            # Simple transposed convolution without complex attention
            output_k = F.conv_transpose1d(
                x, kernel_k, bias_k,
                stride=self.stride, padding=self.padding,
                output_padding=self.output_padding, dilation=self.dilation,
                groups=self.groups
            )
            
            # Weight by kernel attention
            output_k = output_k * kernel_attn[:, k:k+1, :]
            outputs.append(output_k)
        
        # Combine all kernel outputs
        output = sum(outputs)
        
        return output 