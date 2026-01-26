"""
Model Evaluator

Provides unified evaluation pipeline for CoVALend model.
"""

import torch
import numpy as np
import json
from typing import Dict, Optional, List, Any
from tqdm import tqdm

from .metrics import CreditScoringMetrics, FairnessMetrics


class ModelEvaluator:
    """
    Unified model evaluator for CoVALend
    
    Handles evaluation of credit scoring models with fairness metrics.
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        """
        Initialize evaluator
        
        Args:
            model: CoVALend model
            tokenizer: Tokenizer
            device (str): Device to run evaluation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.model.to(device)
        self.model.eval()
    
    def evaluate(
        self,
        test_data: List[Dict],
        batch_size: int = 8,
        compute_fairness: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data
        
        Args:
            test_data (List[Dict]): Test dataset
            batch_size (int): Batch size for evaluation
            compute_fairness (bool): Whether to compute fairness metrics
            
        Returns:
            Dict containing all metrics
        """
        all_preds = []
        all_labels = []
        all_probs = []
        all_groups = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
                batch = test_data[i:i+batch_size]
                
                # Prepare inputs
                texts = [item['input'] for item in batch]
                labels = [item.get('label', item.get('output', 0)) for item in batch]
                groups = [item.get('group', 0) for item in batch]
                
                # Tokenize
                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Get predictions
                outputs = self.model.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_probabilities=True
                )
                
                preds = outputs['predictions'].cpu().numpy()
                probs = outputs['probabilities'][:, 1].cpu().numpy()  # Probability of positive class
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_probs.extend(probs)
                all_groups.extend(groups)
        
        # Convert to arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_groups = np.array(all_groups)
        
        # Calculate performance metrics
        perf_metrics = CreditScoringMetrics.calculate_all_metrics(
            all_labels, all_preds, all_probs
        )
        
        # Calculate fairness metrics if requested
        fairness_metrics = {}
        if compute_fairness and len(np.unique(all_groups)) > 1:
            fairness_metrics = FairnessMetrics.calculate_all_fairness_metrics(
                all_labels, all_preds, all_groups
            )
        
        # Combine results
        results = {
            'performance': perf_metrics,
            'fairness': fairness_metrics,
            'predictions': all_preds.tolist(),
            'probabilities': all_probs.tolist(),
            'labels': all_labels.tolist(),
            'groups': all_groups.tolist()
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results"""
        CreditScoringMetrics.print_metrics(results['performance'], "Performance Metrics")
        
        if results['fairness']:
            FairnessMetrics.print_fairness_metrics(results['fairness'], "Fairness Metrics")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON"""
        # Convert numpy types to Python types
        saveable_results = {
            'performance': {k: float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, (np.int32, np.int64)) else v 
                           for k, v in results['performance'].items()},
            'fairness': {k: float(v) for k, v in results['fairness'].items()},
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(saveable_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_path}")
