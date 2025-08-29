import torch
import torch.nn as nn
import torch.nn.functional as F

class GRC_LoRA_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, r=4):
        super().__init__()
        # Ensure groups doesn't exceed in_channels and out_channels
        groups = min(in_channels, out_channels, 4)  # Use smaller of the three
        
        # Ensure minimum channels for groups
        if in_channels < groups:
            in_channels = groups
        if out_channels < groups:
            out_channels = groups
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=(kernel_size - 1) * dilation // 2, 
            dilation=dilation, groups=groups
        )
        self.lora_A = nn.Parameter(torch.randn(in_channels, r))
        self.lora_B = nn.Parameter(torch.randn(r, out_channels))
        self.lora_scaling = nn.Parameter(torch.ones(1))
        self.output_projection = nn.Conv1d(out_channels, out_channels, 1)
        
        # Ensure norm groups doesn't exceed out_channels
        norm_groups = min(8, out_channels // 4) if out_channels >= 4 else 1
        self.norm = nn.GroupNorm(norm_groups, out_channels)
        self.activation = nn.SiLU()
        
    def forward(self, x):
        base_output = self.conv(x)
        
        # Ensure LoRA adaptation output matches base_output dimensions
        # Get input and output dimensions
        batch_size, in_channels, time_steps = x.shape
        _, out_channels, _ = base_output.shape
        
        # Create LoRA adaptation matrix
        lora_adaptation = torch.matmul(self.lora_A, self.lora_B)  # (in_channels, out_channels)
        
        # Apply LoRA adaptation properly
        # Reshape x: (B, in_channels, T) -> (B*T, in_channels)
        x_reshaped = x.transpose(1, 2).reshape(-1, in_channels)
        
        # Apply LoRA: (B*T, in_channels) @ (in_channels, out_channels) -> (B*T, out_channels)
        lora_output = torch.matmul(x_reshaped, lora_adaptation)
        
        # Reshape back: (B*T, out_channels) -> (B, out_channels, T)
        lora_output = lora_output.reshape(batch_size, time_steps, out_channels).transpose(1, 2)
        
        # Combine base output with LoRA adaptation
        lora_combined = base_output + self.lora_scaling * lora_output
        
        # Process through output projection
        output = self.output_projection(lora_combined)
        output = self.norm(output)
        output = self.activation(output)
        
        # Ensure residual connection dimensions match
        if x.size(1) != output.size(1):
            # Use 1x1 conv to match dimensions if needed
            if not hasattr(self, 'residual_proj'):
                self.residual_proj = nn.Conv1d(x.size(1), output.size(1), 1)
            x = self.residual_proj(x)
        
        return output + x

class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        
        # Project condition to match feature dimensions
        self.condition_projection = nn.Linear(condition_dim, feature_dim * 2)  # gamma and beta
        
    def forward(self, features, speaker_emb=None, emotion_emb=None):
        # FiLM: Feature-wise Linear Modulation
        # Combine speaker and emotion embeddings if both are provided
        if speaker_emb is not None and emotion_emb is not None:
            condition = torch.cat([speaker_emb, emotion_emb], dim=1)
        elif speaker_emb is not None:
            condition = speaker_emb
        elif emotion_emb is not None:
            condition = emotion_emb
        else:
            return features  # No conditioning
            
        # Get actual feature dimensions from input
        actual_feature_dim = features.size(1)
        
        # Ensure condition has the right dimensions
        if condition.size(1) != self.condition_projection.in_features:
            # Pad or truncate condition to match expected dimensions
            if condition.size(1) < self.condition_projection.in_features:
                # Pad with zeros
                padding = torch.zeros(condition.size(0), 
                                    self.condition_projection.in_features - condition.size(1),
                                    device=condition.device)
                condition = torch.cat([condition, padding], dim=1)
            else:
                # Truncate
                condition = condition[:, :self.condition_projection.in_features]
            
        # Project condition
        projected_condition = self.condition_projection(condition)
        
        # Split into gamma and beta
        gamma, beta = torch.chunk(projected_condition, 2, dim=1)
        
        # Ensure gamma and beta match actual feature dimensions
        if gamma.size(1) != actual_feature_dim:
            if gamma.size(1) > actual_feature_dim:
                # Truncate to match
                gamma = gamma[:, :actual_feature_dim]
                beta = beta[:, :actual_feature_dim]
            else:
                # Pad to match
                padding_size = actual_feature_dim - gamma.size(1)
                gamma = F.pad(gamma, (0, padding_size), 'constant', 1.0)
                beta = F.pad(beta, (0, padding_size), 'constant', 0.0)
        
        # Apply modulation (broadcast properly)
        gamma = gamma.unsqueeze(-1)  # (B, actual_feature_dim, 1)
        beta = beta.unsqueeze(-1)    # (B, actual_feature_dim, 1)
        
        return features * gamma + beta

class MultiReceptiveFieldBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 3, 5], groups=4, r=16, dropout=0.1):
        super().__init__()
        # Ensure out_channels is divisible by number of dilations
        channels_per_dilation = out_channels // len(dilations)
        # Ensure channels_per_dilation is divisible by groups
        channels_per_dilation = (channels_per_dilation // groups) * groups
        
        # Ensure minimum channels per dilation
        if channels_per_dilation < groups:
            channels_per_dilation = groups
        
        self.conv_layers = nn.ModuleList([
            GRC_LoRA_Block(in_channels, channels_per_dilation, 3, dilation, r)
            for dilation in dilations
        ])
        
        # Adjust final output to match expected out_channels
        total_channels = channels_per_dilation * len(dilations)
        self.fusion = nn.Conv1d(total_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Ensure groups doesn't exceed out_channels for normalization
        norm_groups = min(8, out_channels // 4) if out_channels >= 4 else 1
        self.norm = nn.GroupNorm(norm_groups, out_channels)
        
    def forward(self, x, speaker_emb=None, emotion_emb=None):
        outputs = [conv(x) for conv in self.conv_layers]
        concatenated = torch.cat(outputs, dim=1)
        output = self.fusion(concatenated)
        output = self.norm(output)
        output = self.dropout(output)
        return output + x 