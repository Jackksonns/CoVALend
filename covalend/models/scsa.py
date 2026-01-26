"""
SCSA (Sensitive Category Suppression Attention) Module

Implements a lightweight attention module for fairness enhancement in credit scoring.
Position: between post-attention LayerNorm and MLP in the LLaMA architecture.

Reference:
    Based on the CoVALend paper and SCSA research:
    Si, Y., et al. (2025). "SCSA: Exploring the synergistic effects between 
    spatial and channel attention." Neurocomputing, 634, 129866.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict


class SCSA(nn.Module):
    """
    Sensitive Category Suppression Attention Module
    
    This module uses multi-scale 1D convolutions to capture local and global
    information, combined with channel attention mechanisms to improve
    representational selectivity and fairness in credit scoring models.
    
    Args:
        dim (int): Hidden dimension of the model
        head_num (int): Number of attention heads
        window_size (int): Window size for local processing
        group_kernel_sizes (List[int]): Multi-scale kernel sizes [3,5,7,9]
        qkv_bias (bool): Whether to use bias in QKV projections
        fuse_bn (bool): Whether to fuse batch normalization
        down_sample_mode (str): Downsampling mode ('avg_pool')
        attn_drop_ratio (float): Attention dropout ratio
        gate_layer (str): Gate layer type ('sigmoid' or 'softmax')
        use_bnb (bool): Whether to use bitsandbytes quantized layers
        
    Input:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)
        
    Output:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
    """
    
    def __init__(
        self,
        dim: int,
        head_num: int,
        window_size: int = 7,
        group_kernel_sizes: List[int] = None,
        qkv_bias: bool = False,
        fuse_bn: bool = False,
        down_sample_mode: str = 'avg_pool',
        attn_drop_ratio: float = 0.1,
        gate_layer: str = 'sigmoid',
        use_bnb: bool = False,
    ):
        super(SCSA, self).__init__()
        
        if group_kernel_sizes is None:
            group_kernel_sizes = [3, 5, 7, 9]
            
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        # Ensure dimension is divisible by 4 for group processing
        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = self.dim // 4

        # Multi-scale 1D convolutions (adapted for sequence data)
        self.local_dwc = nn.Conv1d(
            self.group_chans, self.group_chans, 
            kernel_size=group_kernel_sizes[0],
            padding=group_kernel_sizes[0] // 2, 
            groups=self.group_chans
        )
        self.global_dwc_s = nn.Conv1d(
            self.group_chans, self.group_chans, 
            kernel_size=group_kernel_sizes[1],
            padding=group_kernel_sizes[1] // 2, 
            groups=self.group_chans
        )
        self.global_dwc_m = nn.Conv1d(
            self.group_chans, self.group_chans, 
            kernel_size=group_kernel_sizes[2],
            padding=group_kernel_sizes[2] // 2, 
            groups=self.group_chans
        )
        self.global_dwc_l = nn.Conv1d(
            self.group_chans, self.group_chans, 
            kernel_size=group_kernel_sizes[3],
            padding=group_kernel_sizes[3] // 2, 
            groups=self.group_chans
        )

        # Attention gating
        self.sa_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.LayerNorm(dim)  # Use LayerNorm instead of GroupNorm
        self.norm = nn.LayerNorm(dim)  # Channel normalization

        # Simplified Q, K, V linear layers
        if use_bnb:
            try:
                import bitsandbytes as bnb
                self.q = bnb.nn.Linear8bitLt(dim, dim, bias=qkv_bias)
                self.k = bnb.nn.Linear8bitLt(dim, dim, bias=qkv_bias)
                self.v = bnb.nn.Linear8bitLt(dim, dim, bias=qkv_bias)
            except ImportError:
                # Fallback to regular linear layers
                self.q = nn.Linear(dim, dim, bias=qkv_bias)
                self.k = nn.Linear(dim, dim, bias=qkv_bias)
                self.v = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)

        # Attention dropout
        self.attn_dropout = nn.Dropout(attn_drop_ratio)
        
        # Channel attention gating
        self.ca_gate = nn.Softmax(dim=-1) if gate_layer == 'softmax' else nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale processing
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C) - Llama model hidden states format
            
        Returns:
            torch.Tensor: Output tensor of shape (B, L, C)
        """
        # Get input dimensions
        b, l, c = x.size()  # batch_size, sequence_length, hidden_size
        
        # Convert (B, L, C) to (B, C, L) for 1D convolution processing
        x_conv = x.transpose(1, 2)  # (B, C, L)
        
        # Group processing
        group_chans = c // 4
        l_x, g_x_s, g_x_m, g_x_l = torch.split(x_conv, group_chans, dim=1)
        
        # Multi-scale 1D convolution processing
        l_out = self.local_dwc(l_x)
        g_s_out = self.global_dwc_s(g_x_s)
        g_m_out = self.global_dwc_m(g_x_m)
        g_l_out = self.global_dwc_l(g_x_l)
        
        # Combine multi-scale features
        combined = torch.cat([l_out, g_s_out, g_m_out, g_l_out], dim=1)
        
        # Convert back to (B, L, C) format for normalization
        combined = combined.transpose(1, 2)  # (B, L, C)
        
        # Apply normalization and gating
        normalized = self.norm_h(combined)
        gated = self.sa_gate(normalized)
        
        # Apply gating to original input
        x = x * gated
        
        # Channel attention modeling
        # Use sequence average pooling
        y = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        
        # Apply channel normalization
        y = self.norm(y)
        
        # Simplified channel attention
        q = self.q(y)  # (B, 1, C)
        k = self.k(y)  # (B, 1, C)
        v = self.v(y)  # (B, 1, C)
        
        # Calculate attention weights
        attn_weights = torch.softmax((q @ k.transpose(-2, -1)) * self.scaler, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = attn_weights @ v  # (B, 1, C)
        
        # Apply gating
        attn_output = self.ca_gate(attn_output)
        
        # Apply channel attention to entire sequence
        x = x * attn_output
        
        return x

    @staticmethod
    def calculate_fairness_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        groups: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate fairness metrics (EOD, AOD, DI) based on project implementation
        
        Metrics:
        - EOD (Equal Opportunity Difference): |TPR(G=1) - TPR(G=0)|
        - AOD (Average Odds Difference): 0.5 * [|FPR(G=1) - FPR(G=0)| + |TPR(G=1) - TPR(G=0)|]
        - DI (Disparate Impact): min(P(Ŷ=1|G=1) / P(Ŷ=1|G=0), P(Ŷ=1|G=0) / P(Ŷ=1|G=1))
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            groups (np.ndarray): Group identifiers (0 or 1)
            
        Returns:
            Dict[str, float]: Dictionary containing fairness metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        groups = np.array(groups)
        
        metrics = {}
        
        # Equal Opportunity Difference (EOD)
        mask1_pos = (groups == 1) & (y_true == 1)
        mask0_pos = (groups == 0) & (y_true == 1)
        
        if mask1_pos.sum() > 0 and mask0_pos.sum() > 0:
            tpr1 = (y_pred[mask1_pos] == 1).mean()
            tpr0 = (y_pred[mask0_pos] == 1).mean()
            metrics['EOD'] = abs(tpr1 - tpr0)
        else:
            metrics['EOD'] = 0.0
        
        # Average Odds Difference (AOD)
        mask1_neg = (groups == 1) & (y_true == 0)
        mask0_neg = (groups == 0) & (y_true == 0)
        
        if mask1_neg.sum() > 0 and mask0_neg.sum() > 0:
            fpr1 = (y_pred[mask1_neg] == 1).mean()
            fpr0 = (y_pred[mask0_neg] == 1).mean()
            aod = 0.5 * (abs(fpr1 - fpr0) + metrics.get('EOD', 0))
            metrics['AOD'] = aod
        else:
            metrics['AOD'] = 0.0
        
        # Disparate Impact (DI)
        pred_rate_1 = (y_pred[groups == 1] == 1).mean() if (groups == 1).sum() > 0 else 0
        pred_rate_0 = (y_pred[groups == 0] == 1).mean() if (groups == 0).sum() > 0 else 1e-8
        
        if pred_rate_0 > 1e-8 and pred_rate_1 > 0:
            metrics['DI'] = min(pred_rate_1 / pred_rate_0, pred_rate_0 / pred_rate_1)
        else:
            metrics['DI'] = 0.0
        
        return metrics
