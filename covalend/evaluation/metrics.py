"""
Metrics Module

Implements evaluation metrics for credit scoring including:
- Performance metrics: Accuracy, F1, Precision, Recall, MCC
- Cost-aware metrics: Miss Rate (False Negative Rate)
- Fairness metrics: EOD, AOD, Disparate Impact

Based on the CoVALend paper evaluation protocol.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from typing import Dict, Optional, Tuple


class CreditScoringMetrics:
    """
    Credit scoring performance metrics
    
    Implements metrics reported in the CoVALend paper:
    - Accuracy: Overall classification accuracy
    - F1 Score: Harmonic mean of precision and recall
    - Precision: True positives / (True positives + False positives)
    - Recall: True positives / (True positives + False negatives)
    - MCC (Matthews Correlation Coefficient): Balanced measure for imbalanced data
    - Miss Rate: False negatives / (True positives + False negatives)
    - AUC: Area under the ROC curve
    """
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (Optional[np.ndarray]): Prediction probabilities for positive class
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix based metrics
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Miss Rate (False Negative Rate) - critical for credit scoring
        metrics['miss'] = fn / (tp + fn + 1e-8)
        
        # False Positive Rate
        metrics['fpr'] = fp / (fp + tn + 1e-8)
        
        # True Negative Rate (Specificity)
        metrics['tnr'] = tn / (tn + fp + 1e-8)
        
        # AUC if probabilities are provided
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except Exception:
                metrics['auc'] = 0.0
        
        # Confusion matrix components
        metrics['tp'] = int(tp)
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        
        return metrics
    
    @staticmethod
    def calculate_miss_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Miss Rate (False Negative Rate)
        
        Miss = FN / (TP + FN)
        
        This is a critical metric in credit scoring as it represents
        the proportion of actual defaults that are missed by the model.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            float: Miss rate
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        return fn / (tp + fn + 1e-8)
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
        """
        Print metrics in a formatted table
        
        Args:
            metrics (Dict[str, float]): Metrics dictionary
            title (str): Title for the metrics table
        """
        print(f"\n{'='*50}")
        print(f"{title:^50}")
        print(f"{'='*50}")
        
        # Performance metrics
        print("\nPerformance Metrics:")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  F1 Score:  {metrics.get('f1', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  MCC:       {metrics.get('mcc', 0):.4f}")
        
        # Cost-aware metrics
        print("\nCost-Aware Metrics:")
        print(f"  Miss Rate: {metrics.get('miss', 0):.4f}")
        print(f"  FPR:       {metrics.get('fpr', 0):.4f}")
        print(f"  TNR:       {metrics.get('tnr', 0):.4f}")
        
        if 'auc' in metrics:
            print(f"  AUC:       {metrics.get('auc', 0):.4f}")
        
        # Confusion matrix
        if all(k in metrics for k in ['tp', 'tn', 'fp', 'fn']):
            print("\nConfusion Matrix:")
            print(f"  TP: {metrics['tp']:4d}  FN: {metrics['fn']:4d}")
            print(f"  FP: {metrics['fp']:4d}  TN: {metrics['tn']:4d}")
        
        print(f"{'='*50}\n")


class FairnessMetrics:
    """
    Fairness metrics for algorithmic fairness evaluation
    
    Implements metrics from the CoVALend paper:
    - EOD (Equal Opportunity Difference): |TPR(G=1) - TPR(G=0)|
    - AOD (Average Odds Difference): 0.5 * [|FPR(G=1) - FPR(G=0)| + |TPR(G=1) - TPR(G=0)|]
    - DI (Disparate Impact): min(P(Ŷ=1|G=1) / P(Ŷ=1|G=0), P(Ŷ=1|G=0) / P(Ŷ=1|G=1))
    
    Reference:
        Hardt, M., et al. (2016). "Equality of opportunity in supervised learning."
        Feldman, M., et al. (2015). "Certifying and removing disparate impact."
    """
    
    @staticmethod
    def calculate_all_fairness_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all fairness metrics
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            groups (np.ndarray): Group identifiers (0 or 1)
            
        Returns:
            Dict[str, float]: Dictionary of fairness metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        groups = np.array(groups)
        
        metrics = {}
        
        # Equal Opportunity Difference (EOD)
        metrics['eod'] = FairnessMetrics.equal_opportunity_difference(y_true, y_pred, groups)
        
        # Average Odds Difference (AOD)
        metrics['aod'] = FairnessMetrics.average_odds_difference(y_true, y_pred, groups)
        
        # Disparate Impact (DI)
        metrics['di'] = FairnessMetrics.disparate_impact(y_pred, groups)
        
        # Statistical Parity Difference
        metrics['spd'] = FairnessMetrics.statistical_parity_difference(y_pred, groups)
        
        return metrics
    
    @staticmethod
    def equal_opportunity_difference(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray
    ) -> float:
        """
        Calculate Equal Opportunity Difference (EOD)
        
        EOD = |TPR(G=1) - TPR(G=0)|
        
        Where TPR = True Positive Rate = TP / (TP + FN)
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            groups (np.ndarray): Group identifiers
            
        Returns:
            float: Equal Opportunity Difference
        """
        mask1_pos = (groups == 1) & (y_true == 1)
        mask0_pos = (groups == 0) & (y_true == 1)
        
        if mask1_pos.sum() > 0 and mask0_pos.sum() > 0:
            tpr1 = (y_pred[mask1_pos] == 1).mean()
            tpr0 = (y_pred[mask0_pos] == 1).mean()
            return abs(tpr1 - tpr0)
        else:
            return 0.0
    
    @staticmethod
    def average_odds_difference(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray
    ) -> float:
        """
        Calculate Average Odds Difference (AOD)
        
        AOD = 0.5 * [|FPR(G=1) - FPR(G=0)| + |TPR(G=1) - TPR(G=0)|]
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            groups (np.ndarray): Group identifiers
            
        Returns:
            float: Average Odds Difference
        """
        # TPR difference
        eod = FairnessMetrics.equal_opportunity_difference(y_true, y_pred, groups)
        
        # FPR difference
        mask1_neg = (groups == 1) & (y_true == 0)
        mask0_neg = (groups == 0) & (y_true == 0)
        
        if mask1_neg.sum() > 0 and mask0_neg.sum() > 0:
            fpr1 = (y_pred[mask1_neg] == 1).mean()
            fpr0 = (y_pred[mask0_neg] == 1).mean()
            fpr_diff = abs(fpr1 - fpr0)
        else:
            fpr_diff = 0.0
        
        return 0.5 * (fpr_diff + eod)
    
    @staticmethod
    def disparate_impact(y_pred: np.ndarray, groups: np.ndarray) -> float:
        """
        Calculate Disparate Impact (DI)
        
        DI = min(P(Ŷ=1|G=1) / P(Ŷ=1|G=0), P(Ŷ=1|G=0) / P(Ŷ=1|G=1))
        
        A value close to 1.0 indicates fairness.
        The 80% rule suggests DI >= 0.8 is acceptable.
        
        Args:
            y_pred (np.ndarray): Predicted labels
            groups (np.ndarray): Group identifiers
            
        Returns:
            float: Disparate Impact
        """
        pred_rate_1 = (y_pred[groups == 1] == 1).mean() if (groups == 1).sum() > 0 else 0
        pred_rate_0 = (y_pred[groups == 0] == 1).mean() if (groups == 0).sum() > 0 else 1e-8
        
        if pred_rate_0 > 1e-8 and pred_rate_1 > 0:
            return min(pred_rate_1 / pred_rate_0, pred_rate_0 / pred_rate_1)
        else:
            return 0.0
    
    @staticmethod
    def statistical_parity_difference(y_pred: np.ndarray, groups: np.ndarray) -> float:
        """
        Calculate Statistical Parity Difference
        
        SPD = |P(Ŷ=1|G=1) - P(Ŷ=1|G=0)|
        
        Args:
            y_pred (np.ndarray): Predicted labels
            groups (np.ndarray): Group identifiers
            
        Returns:
            float: Statistical Parity Difference
        """
        pred_rate_1 = (y_pred[groups == 1] == 1).mean() if (groups == 1).sum() > 0 else 0
        pred_rate_0 = (y_pred[groups == 0] == 1).mean() if (groups == 0).sum() > 0 else 0
        
        return abs(pred_rate_1 - pred_rate_0)
    
    @staticmethod
    def print_fairness_metrics(metrics: Dict[str, float], title: str = "Fairness Metrics"):
        """
        Print fairness metrics in a formatted table
        
        Args:
            metrics (Dict[str, float]): Fairness metrics dictionary
            title (str): Title for the metrics table
        """
        print(f"\n{'='*50}")
        print(f"{title:^50}")
        print(f"{'='*50}")
        
        print("\nFairness Metrics:")
        print(f"  EOD (Equal Opportunity Diff):  {metrics.get('eod', 0):.4f}")
        print(f"  AOD (Average Odds Diff):       {metrics.get('aod', 0):.4f}")
        print(f"  DI (Disparate Impact):         {metrics.get('di', 0):.4f}")
        print(f"  SPD (Statistical Parity Diff): {metrics.get('spd', 0):.4f}")
        
        # Interpretation
        print("\nInterpretation:")
        di = metrics.get('di', 0)
        if di >= 0.8:
            print(f"  ✓ DI >= 0.8 (Acceptable by 80% rule)")
        else:
            print(f"  ✗ DI < 0.8 (May indicate bias)")
        
        eod = metrics.get('eod', 0)
        if eod < 0.1:
            print(f"  ✓ EOD < 0.1 (Good equal opportunity)")
        else:
            print(f"  ✗ EOD >= 0.1 (Potential equal opportunity gap)")
        
        print(f"{'='*50}\n")
