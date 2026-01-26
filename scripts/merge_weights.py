import os
import sys
import argparse
import logging
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights to base LLaMA model")
    parser.add_argument("--base_model_name", type=str, required=True, help="Base model name or path")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged model")
    args = parser.parse_args()

    logger.info(f"Loading base model: {args.base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    logger.info(f"Loading LoRA adapters: {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    logger.info("Merging weights...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    logger.info("Done.")

if __name__ == "__main__":
    main()

