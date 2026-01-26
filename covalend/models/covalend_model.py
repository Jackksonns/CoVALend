"""
CoVALend Model Module

Main model architecture combining LLaMA backbone with Finance Encoder and SCSA.

Reference:
    "CoVALend: A Value-Alignment Pipeline for Compliant Micro-Lending 
    with Minimal-Intrusion Tabular-Text Fusion"
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json
from typing import Tuple, Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .finance_encoder import FinanceEncoder
from .scsa import SCSA
from .config import CovalendConfig


class CovalendModel(nn.Module):
    """
    CoVALend Model: LLaMA backbone with Finance Encoder and SCSA
    
    This model combines:
    1. LLaMA-2-7B as the base language model
    2. Finance Encoder for tabular-text fusion
    3. SCSA module for fairness-aware representation learning
    4. Classification head for binary credit default prediction
    
    Args:
        config (CovalendConfig): Model configuration
        base_model (Optional): Pre-loaded base model
        tokenizer (Optional): Pre-loaded tokenizer
        
    Input:
        input_ids (torch.Tensor): Input token IDs
        attention_mask (torch.Tensor): Attention mask
        structured_features (Optional[Dict]): Structured financial features
        labels (Optional[torch.Tensor]): Labels for training
        
    Output:
        Dict containing logits, loss (if labels provided), and optional fairness metrics
    """
    
    def __init__(
        self, 
        config: CovalendConfig,
        base_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        super(CovalendModel, self).__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.use_finance_encoder = config.use_finance_encoder
        self.use_scsa = config.use_scsa
        
        # Load or use provided base LLaMA model
        if base_model is not None:
            self.base_model = base_model
            self.tokenizer = tokenizer
            if hasattr(base_model, 'config'):
                self.hidden_size = base_model.config.hidden_size
                self.num_attention_heads = base_model.config.num_attention_heads
        else:
            try:
                print(f"Loading base model: {config.base_model_name}")
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
                
                # Update config with actual values
                self.hidden_size = self.base_model.config.hidden_size
                self.num_attention_heads = self.base_model.config.num_attention_heads
                
                print(f"Model loaded successfully. Hidden size: {self.hidden_size}")
                
            except Exception as e:
                print(f"Warning: Could not load base model {config.base_model_name}: {e}")
                print("Model will be initialized without base model for demonstration.")
                self.base_model = None
                self.tokenizer = None
        
        # Finance Encoder for structured feature fusion
        if self.use_finance_encoder:
            self.finance_encoder = FinanceEncoder(
                hidden_size=self.hidden_size,
                numeric_features=config.structured_numeric_features,
                categorical_features=config.structured_categorical_features,
                embed_dim=config.structured_embed_dim,
                cat_embed_dim=config.structured_cat_embed_dim,
                dropout=config.dropout
            )
            print("Finance Encoder initialized")
        else:
            self.finance_encoder = None
        
        # SCSA module for fairness enhancement
        if self.use_scsa:
            self.scsa = SCSA(
                dim=self.hidden_size,
                head_num=self.num_attention_heads,
                window_size=config.scsa_window_size,
                group_kernel_sizes=config.scsa_kernel_sizes,
                qkv_bias=config.scsa_qkv_bias,
                fuse_bn=False,
                down_sample_mode='avg_pool',
                attn_drop_ratio=config.scsa_attn_drop_ratio,
                gate_layer=config.scsa_gate_layer,
                use_bnb=False
            )
            print("SCSA module initialized")
        else:
            self.scsa = None
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size // 2, config.num_labels)
        )
        
        # Initialize weights
        self._init_weights()
        print("CoVALend model initialization complete")
    
    def _init_weights(self):
        """Initialize weights for new modules"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        structured_features: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional feature fusion and SCSA processing
        
        Args:
            input_ids (torch.Tensor): Input token IDs (B, L)
            attention_mask (Optional[torch.Tensor]): Attention mask (B, L)
            structured_features (Optional[Dict]): Structured financial features
            labels (Optional[torch.Tensor]): Labels for computing loss (B,)
            
        Returns:
            Dict containing:
                - logits: Classification logits (B, num_labels)
                - loss: Cross-entropy loss (if labels provided)
                - hidden_states: Last layer hidden states (B, L, C)
        """
        if self.base_model is None:
            # Dummy forward pass for demonstration
            batch_size = input_ids.shape[0]
            dummy_hidden = torch.randn(
                batch_size, 1, self.hidden_size, 
                device=input_ids.device, 
                dtype=torch.float32
            )
            logits = self.classifier(dummy_hidden.squeeze(1))
            
            output_dict = {'logits': logits}
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                output_dict['loss'] = loss
            
            return output_dict
        
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # Extract hidden states from the last layer
        hidden_states = base_outputs.hidden_states[-1]  # (B, L, C)
        
        # Apply Finance Encoder if enabled
        if self.finance_encoder is not None and structured_features is not None:
            hidden_states = self.finance_encoder(hidden_states, structured_features)
        
        # Apply SCSA if enabled
        if self.scsa is not None:
            hidden_states = self.scsa(hidden_states)
        
        # Use the first token for classification (CLS token equivalent)
        cls_hidden = hidden_states[:, 0, :]  # (B, C)
        
        # Classification
        logits = self.classifier(cls_hidden)  # (B, num_labels)
        
        # Prepare output
        output_dict = {
            'logits': logits,
            'hidden_states': hidden_states
        }
        
        # Compute loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            output_dict['loss'] = loss
        
        return output_dict
    
    def predict(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        structured_features: Optional[Dict[str, torch.Tensor]] = None,
        return_probabilities: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions without computing gradients
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (Optional[torch.Tensor]): Attention mask
            structured_features (Optional[Dict]): Structured financial features
            return_probabilities (bool): Whether to return probabilities
            
        Returns:
            Dict containing predictions and optional probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, structured_features)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            
            result = {'predictions': predictions}
            
            if return_probabilities:
                probabilities = torch.softmax(logits, dim=-1)
                result['probabilities'] = probabilities
            
            return result
    
    def predict_with_fairness(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        structured_features: Optional[Dict[str, torch.Tensor]] = None,
        groups: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Make predictions and compute fairness metrics
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (Optional[torch.Tensor]): Attention mask
            structured_features (Optional[Dict]): Structured financial features
            groups (Optional[torch.Tensor]): Group identifiers for fairness metrics
            labels (Optional[torch.Tensor]): True labels for fairness computation
            
        Returns:
            Dict containing predictions, probabilities, and fairness metrics
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, structured_features)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions
        }
        
        # Calculate fairness metrics if groups and labels are provided
        if groups is not None and labels is not None and self.scsa is not None:
            fairness_metrics = SCSA.calculate_fairness_metrics(
                y_true=labels.cpu().numpy(),
                y_pred=predictions.cpu().numpy(),
                groups=groups.cpu().numpy()
            )
            result['fairness_metrics'] = fairness_metrics
        
        return result
    
    def save_pretrained(self, save_directory: str):
        """
        Save model and configuration
        
        Args:
            save_directory (str): Directory to save the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save base model if exists
        if self.base_model is not None:
            self.base_model.save_pretrained(save_directory)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)
        
        # Save configuration
        self.config.save_pretrained(save_directory)
        
        # Save custom components
        components = {
            'finance_encoder_state_dict': self.finance_encoder.state_dict() if self.finance_encoder else None,
            'scsa_state_dict': self.scsa.state_dict() if self.scsa else None,
            'classifier_state_dict': self.classifier.state_dict(),
        }
        
        torch.save(components, os.path.join(save_directory, "covalend_components.pt"))
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls, 
        save_directory: str,
        device: Optional[str] = None,
        **kwargs
    ) -> "CovalendModel":
        """
        Load model from saved directory
        
        Args:
            save_directory (str): Directory containing the saved model
            device (Optional[str]): Device to load the model on
            
        Returns:
            CovalendModel: Loaded model
        """
        # Load configuration
        config = CovalendConfig.from_pretrained(save_directory)
        
        # Load base model
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                save_directory,
                torch_dtype=torch.float16,
                device_map="auto" if device is None else device,
                **kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(save_directory)
            print("Base model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load base model: {e}")
            base_model = None
            tokenizer = None
        
        # Initialize model
        model = cls(config=config, base_model=base_model, tokenizer=tokenizer)
        
        # Load custom components
        components_path = os.path.join(save_directory, "covalend_components.pt")
        if os.path.exists(components_path):
            components = torch.load(components_path, map_location='cpu')
            
            if model.finance_encoder and components['finance_encoder_state_dict']:
                model.finance_encoder.load_state_dict(components['finance_encoder_state_dict'])
                print("Finance Encoder loaded")
            
            if model.scsa and components['scsa_state_dict']:
                model.scsa.load_state_dict(components['scsa_state_dict'])
                print("SCSA module loaded")
            
            if components['classifier_state_dict']:
                model.classifier.load_state_dict(components['classifier_state_dict'])
                print("Classifier loaded")
        
        print(f"Model loaded from {save_directory}")
        return model
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (for LoRA setup)"""
        trainable_params = []
        
        # Finance Encoder parameters
        if self.finance_encoder:
            trainable_params.extend(self.finance_encoder.parameters())
        
        # SCSA parameters
        if self.scsa:
            trainable_params.extend(self.scsa.parameters())
        
        # Classifier parameters
        trainable_params.extend(self.classifier.parameters())
        
        return trainable_params
    
    def print_trainable_parameters(self):
        """Print the number of trainable parameters"""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {100 * trainable_params / all_params:.2f}%"
        )
