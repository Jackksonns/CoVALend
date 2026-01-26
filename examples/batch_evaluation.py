"""
Batch evaluation example for CoVALend model
"""

import torch
import json
from transformers import AutoTokenizer
from covalend import CovalendModel, CovalendConfig
from covalend.evaluation import ModelEvaluator


def main():
    """Run batch evaluation example"""
    
    # Configuration
    config = CovalendConfig(
        base_model_name="meta-llama/Llama-2-7b-hf",  # Change to your model path
        use_finance_encoder=True,
        use_scsa=True
    )
    
    # Initialize model
    print("Loading model...")
    model = CovalendModel(config)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    
    # Load test data
    print("Loading test data...")
    with open("data/test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Initialize evaluator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = ModelEvaluator(model, tokenizer, device=device)
    
    # Run evaluation
    print("\nEvaluating model...")
    results = evaluator.evaluate(
        test_data=test_data,
        batch_size=8,
        compute_fairness=True
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    evaluator.save_results(results, "outputs/evaluation_results.json")
    print("\nResults saved to outputs/evaluation_results.json")


if __name__ == "__main__":
    main()
