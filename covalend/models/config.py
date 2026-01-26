"""
Configuration for CoVALend Model

This module defines the configuration class for the CoVALend model,
managing hyperparameters and model architecture settings.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class CovalendConfig:
    """
    Configuration class for CoVALend model
    
    Attributes:
        base_model_name (str): Name or path of the base LLaMA model
        hidden_size (int): Hidden size of the model
        num_attention_heads (int): Number of attention heads
        dropout (float): Dropout rate
        use_finance_encoder (bool): Whether to use Finance Encoder
        use_scsa (bool): Whether to use SCSA module
        structured_numeric_features (List[str]): List of numeric feature names
        structured_categorical_features (Dict[str, int]): Dictionary of categorical features and their sizes
        structured_embed_dim (int): Embedding dimension for numeric features
        structured_cat_embed_dim (int): Embedding dimension for categorical features
        scsa_window_size (int): Window size for SCSA
        scsa_kernel_sizes (List[int]): Kernel sizes for multi-scale convolutions in SCSA
        scsa_gate_layer (str): Gate layer type for SCSA ('sigmoid' or 'softmax')
        num_labels (int): Number of classification labels
    """
    # Base model configuration
    base_model_name: str = "meta-llama/Llama-2-7b-hf"
    hidden_size: int = 4096
    num_attention_heads: int = 32
    dropout: float = 0.1
    
    # Module flags
    use_finance_encoder: bool = True
    use_scsa: bool = True
    
    # Finance Encoder configuration
    structured_numeric_features: List[str] = field(
        default_factory=lambda: ["income", "overdue_count", "credit_score", "debt_ratio"]
    )
    structured_categorical_features: Dict[str, int] = field(
        default_factory=lambda: {
            "job_type": 10,
            "education": 5,
            "marital_status": 4
        }
    )
    structured_embed_dim: int = 512
    structured_cat_embed_dim: int = 100
    
    # SCSA configuration
    scsa_window_size: int = 7
    scsa_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 9])
    scsa_gate_layer: str = "sigmoid"
    scsa_qkv_bias: bool = False
    scsa_attn_drop_ratio: float = 0.1
    
    # Classification configuration
    num_labels: int = 2
    
    # Training configuration
    max_seq_length: int = 512
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            "base_model_name": self.base_model_name,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "dropout": self.dropout,
            "use_finance_encoder": self.use_finance_encoder,
            "use_scsa": self.use_scsa,
            "structured_numeric_features": self.structured_numeric_features,
            "structured_categorical_features": self.structured_categorical_features,
            "structured_embed_dim": self.structured_embed_dim,
            "structured_cat_embed_dim": self.structured_cat_embed_dim,
            "scsa_window_size": self.scsa_window_size,
            "scsa_kernel_sizes": self.scsa_kernel_sizes,
            "scsa_gate_layer": self.scsa_gate_layer,
            "scsa_qkv_bias": self.scsa_qkv_bias,
            "scsa_attn_drop_ratio": self.scsa_attn_drop_ratio,
            "num_labels": self.num_labels,
            "max_seq_length": self.max_seq_length,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "CovalendConfig":
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def save_pretrained(self, save_directory: str):
        """Save configuration to directory"""
        import json
        import os
        
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "covalend_config.json")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_pretrained(cls, save_directory: str) -> "CovalendConfig":
        """Load configuration from directory"""
        import json
        import os
        
        config_path = os.path.join(save_directory, "covalend_config.json")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
