"""
Simple inference example for CoVALend model
"""

import torch
from transformers import AutoTokenizer
from covalend import CovalendModel, CovalendConfig


def main():
    """Run simple inference example"""
    
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
    
    # Set to evaluation mode
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Example input
    text = """
    Loan application: Applicant has annual income of $50,000, credit score of 720,
    no overdue payments in the last 12 months. Employment history: 5 years with current employer.
    Loan amount requested: $10,000 for home improvement. Debt-to-income ratio: 0.25.
    """
    
    # Optional: structured features
    structured_features = {
        "income": torch.tensor([50000.0], device=device),
        "credit_score": torch.tensor([720.0], device=device),
        "overdue_count": torch.tensor([0.0], device=device),
        "debt_ratio": torch.tensor([0.25], device=device)
    }
    
    # Tokenize
    print("\nMaking prediction...")
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        result = model.predict(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            structured_features=structured_features,
            return_probabilities=True
        )
    
    # Display results
    prediction = result['predictions'].item()
    probability = result['probabilities'][0][1].item()
    
    print("\n" + "="*50)
    print("Prediction Results")
    print("="*50)
    print(f"Prediction: {'Default' if prediction == 1 else 'No Default'}")
    print(f"Confidence (P(Default)): {probability:.4f}")
    print(f"Risk Level: {get_risk_level(probability)}")
    print("="*50)


def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"


if __name__ == "__main__":
    main()
