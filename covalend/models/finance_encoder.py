"""
Finance Encoder Module

Implements first token fusion with missing value gating for tabular-text fusion.
This module integrates structured financial features into LLaMA's first token embedding.

Reference:
    Based on the CoVALend paper implementation
    "CoVALend: A Value-Alignment Pipeline for Compliant Micro-Lending 
    with Minimal-Intrusion Tabular-Text Fusion"
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List


class FinanceEncoder(nn.Module):
    """
    Finance Encoder with first token fusion and missing value gating
    
    This module processes structured financial features (numeric and categorical)
    and fuses them with the first token of LLaMA's input embeddings through
    a learnable gating mechanism that handles missing values.
    
    Args:
        hidden_size (int): Hidden size from LLaMA config
        numeric_features (List[str]): List of numeric feature names
        categorical_features (Dict[str, int]): Dict of categorical features and their sizes
        embed_dim (int): Numeric feature mapping dimension
        cat_embed_dim (int): Categorical feature embedding dimension
        dropout (float): Dropout rate
        
    Input:
        hidden_states (torch.Tensor): LLaMA hidden states of shape (batch_size, seq_len, hidden_size)
        structured_features (Optional[Dict]): Structured financial features
        
    Output:
        torch.Tensor: Enhanced hidden states of shape (batch_size, seq_len, hidden_size)
    """
    
    def __init__(
        self,
        hidden_size: int,
        numeric_features: List[str],
        categorical_features: Dict[str, int],
        embed_dim: int = 512,
        cat_embed_dim: int = 100,
        dropout: float = 0.1
    ):
        super(FinanceEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.embed_dim = embed_dim
        self.cat_embed_dim = cat_embed_dim
        
        # Numeric feature linear layers and LayerNorm
        self.fin_num_linears = nn.ModuleDict({
            k: nn.Linear(1, embed_dim) for k in numeric_features
        })
        self.fin_num_norms = nn.ModuleDict({
            k: nn.LayerNorm(embed_dim) for k in numeric_features
        })
        
        # Categorical feature embeddings
        self.fin_cat_embeds = nn.ModuleDict({
            k: nn.Embedding(v, cat_embed_dim) for k, v in categorical_features.items()
        })
        self.fin_cat_norms = nn.ModuleDict({
            k: nn.LayerNorm(cat_embed_dim) for k in categorical_features
        })
        
        # Fusion layers
        total_cat_dim = cat_embed_dim * len(categorical_features)
        self.fin_fuse_linear_text = nn.Linear(hidden_size, embed_dim)
        self.fin_fuse_linear_struct = nn.Linear(
            embed_dim + total_cat_dim, 
            embed_dim
        )
        self.fin_fuse_norm = nn.LayerNorm(embed_dim)
        
        # Missing value gate (learnable parameter)
        self.fin_missing_gate = nn.Parameter(torch.randn(embed_dim))
        
        # Final projection back to hidden size
        self.final_projection = nn.Linear(embed_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        structured_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass with structured feature fusion
        
        Args:
            hidden_states (torch.Tensor): LLaMA hidden states (B, L, C)
            structured_features (Optional[Dict]): Structured financial features
            
        Returns:
            torch.Tensor: Enhanced hidden states (B, L, C)
        """
        if structured_features is None:
            return hidden_states
            
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Process numeric features
        num_vecs = []
        for k in self.numeric_features:
            v = structured_features.get(k, None)
            if v is None or (isinstance(v, torch.Tensor) and (v == -100).all()):
                # Use missing gate for missing values
                missing_vec = self.fin_missing_gate.unsqueeze(0).expand(batch_size, -1)
                num_vecs.append(missing_vec)
            else:
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, device=device)
                v = v.view(-1, 1).float()
                out = self.fin_num_linears[k](v)
                out = self.fin_num_norms[k](out)
                num_vecs.append(out)
        
        # Combine numeric features (simple addition)
        if num_vecs:
            num_vec = torch.stack(num_vecs, dim=0).sum(dim=0)  # (B, embed_dim)
        else:
            num_vec = torch.zeros(batch_size, self.embed_dim, device=device)
        
        # Process categorical features
        cat_vecs = []
        for k in self.categorical_features:
            v = structured_features.get(k, None)
            if v is None or (isinstance(v, torch.Tensor) and (v == -100).all()):
                # Use missing gate for missing values
                missing_vec = self.fin_missing_gate[:self.cat_embed_dim].unsqueeze(0).expand(batch_size, -1)
                cat_vecs.append(missing_vec)
            else:
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, device=device)
                v = v.view(-1).long()
                # Clip to valid range
                v = torch.clamp(v, 0, self.categorical_features[k] - 1)
                out = self.fin_cat_embeds[k](v)
                out = self.fin_cat_norms[k](out)
                cat_vecs.append(out)
        
        # Combine categorical features
        if cat_vecs:
            cat_vec = torch.cat(cat_vecs, dim=-1)  # (B, total_cat_dim)
        else:
            cat_vec = torch.zeros(batch_size, 0, device=device)
        
        # Fusion mechanism: combine numeric and categorical features
        struct_vec = torch.cat([num_vec, cat_vec], dim=-1)
        struct_vec = self.fin_fuse_linear_struct(struct_vec)
        struct_vec = self.fin_fuse_norm(struct_vec)
        
        # Project text features to same dimension (use first token)
        text_vec = self.fin_fuse_linear_text(hidden_states[:, 0, :])
        
        # Fuse structured and text features
        fused_vec = struct_vec + text_vec  # (B, embed_dim)
        fused_vec = self.dropout(fused_vec)
        
        # Project back to hidden size
        fused_vec = self.final_projection(fused_vec)  # (B, hidden_size)
        
        # Apply fusion to first token only
        enhanced_states = hidden_states.clone()
        enhanced_states[:, 0, :] = enhanced_states[:, 0, :] + fused_vec
        
        return enhanced_states

    @staticmethod
    def row_to_features(
        row: pd.Series,
        numeric_features: List[str],
        categorical_features: Dict[str, int]
    ) -> Dict[str, np.ndarray]:
        """
        Convert data row to structured features dictionary
        
        Args:
            row (pd.Series): Data row
            numeric_features (List[str]): List of numeric feature names
            categorical_features (Dict[str, int]): Dictionary of categorical features
            
        Returns:
            Dict[str, np.ndarray]: Structured features dictionary
        """
        features = {}
        
        # Extract numeric features
        for feature_name in numeric_features:
            if feature_name in row.index and pd.notna(row[feature_name]):
                try:
                    features[feature_name] = np.array([float(row[feature_name])], dtype=np.float32)
                except (ValueError, TypeError):
                    features[feature_name] = np.array([0.0], dtype=np.float32)
            else:
                features[feature_name] = np.array([-100], dtype=np.float32)  # Missing value indicator
        
        # Extract categorical features
        for feature_name, num_categories in categorical_features.items():
            if feature_name in row.index and pd.notna(row[feature_name]):
                try:
                    # Convert to integer category index
                    cat_value = int(row[feature_name])
                    if 0 <= cat_value < num_categories:
                        features[feature_name] = np.array([cat_value], dtype=np.int64)
                    else:
                        features[feature_name] = np.array([-100], dtype=np.int64)
                except (ValueError, TypeError):
                    features[feature_name] = np.array([-100], dtype=np.int64)
            else:
                features[feature_name] = np.array([-100], dtype=np.int64)  # Missing value indicator
        
        return features

    @staticmethod
    def preprocess_data(
        data: pd.DataFrame,
        numeric_features: List[str],
        categorical_features: Dict[str, int],
        label_column: str = 'TARGET'
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """
        Preprocess data for finance encoder
        
        Args:
            data (pd.DataFrame): Input data
            numeric_features (List[str]): List of numeric feature names
            categorical_features (Dict[str, int]): Dictionary of categorical features
            label_column (str): Name of the label column
            
        Returns:
            Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]: Features and labels
        """
        features_list = []
        labels = None
        
        for _, row in data.iterrows():
            features = FinanceEncoder.row_to_features(row, numeric_features, categorical_features)
            features_list.append(features)
        
        # Convert to batch format
        batch_features = {}
        for feature_name in numeric_features + list(categorical_features.keys()):
            batch_features[feature_name] = np.array([f[feature_name] for f in features_list])
        
        # Extract labels if available
        if label_column in data.columns:
            labels = data[label_column].values
        
        return batch_features, labels
