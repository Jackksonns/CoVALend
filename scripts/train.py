import os
import sys
import yaml
import argparse
import logging
from typing import Dict, Any

import torch
import transformers
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from covalend.models.config import CovalendConfig
from covalend.models.covalend_model import CovalendModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="CoVALend Training Script")
    parser.add_argument("--config", type=str, default="configs/stage1.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs/models", help="Output directory")
    args = parser.parse_args()

    # Load configuration
    config_dict = load_config(args.config)
    model_cfg = config_dict['model']
    train_cfg = config_dict['training']
    
    logger.info("Loading Base Model...")
    
    # Initialize CoVALend Config
    cov_config = CovalendConfig(
        base_model_name=model_cfg['base_model_name'],
        use_finance_encoder=model_cfg['use_finance_encoder'],
        use_scsa=model_cfg['use_scsa'],
        # ... map other config params ...
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['base_model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = CovalendModel(cov_config)
    
    # Prepare for LoRA
    if train_cfg.get('use_lora', True):
        logger.info("Applying LoRA...")
        peft_config = LoraConfig(
            r=train_cfg.get('lora_r', 8),
            lora_alpha=train_cfg.get('lora_alpha', 32),
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # Apply PEFT to the specific sub-modules or the whole model if supported
        # Note: CovalendModel wraps Llama, so we might need to target the backbone
        # For simplicity in this script, we assume standard PEFT usage on the backbone
        model.backbone = get_peft_model(model.backbone, peft_config)
        model.print_trainable_parameters()

    # Load Datasets
    logger.info("Loading Datasets...")
    if 'dataset_path' in train_cfg:
        train_dataset = load_dataset('json', data_files=train_cfg['dataset_path'], split='train')
    else:
        # Dummy dataset for example
        logger.warning("No dataset provided in config, creating dummy dataset.")
        from datasets import Dataset
        train_dataset = Dataset.from_dict({"text": ["Example loan application..."] * 10})

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=train_cfg.get('batch_size', 4),
        gradient_accumulation_steps=train_cfg.get('grad_accum', 4),
        learning_rate=float(train_cfg.get('learning_rate', 2e-4)),
        logging_steps=10,
        num_train_epochs=train_cfg.get('epochs', 3),
        save_steps=100,
        fp16=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train
    logger.info("Starting Training...")
    trainer.train()
    
    # Save
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()

