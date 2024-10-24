from typing import List, Dict, Union
from collections import defaultdict
import numpy as np
from .data_types import QueryResult

class RAGMetrics:
    """Calculates and tracks metrics for RAG performance"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.metrics = defaultdict(list)
        
    def update(self, 
               prediction: str, 
               ground_truth: str, 
               query_result: QueryResult):
        """Update metrics with new prediction"""
        # Calculate exact match
        exact_match = int(prediction.strip().lower() == ground_truth.strip().lower())
        self.metrics['exact_match'].append(exact_match)
        
        # Calculate F1 score
        f1 = self._calculate_f1(prediction, ground_truth)
        self.metrics['f1'].append(f1)
        
        # Calculate accuracy (less strict matching)
        accuracy = int(ground_truth.strip().lower() in prediction.strip().lower())
        self.metrics['accuracy'].append(accuracy)
        
        # Track effective context length
        self.metrics['context_length'].append(query_result.effective_context_length)
        
        # Track number of iterations (for IterDRAG)
        if query_result.sub_queries:
            self.metrics['num_iterations'].append(len(query_result.sub_queries))
            
        # Track confidence
        self.metrics['confidence'].append(query_result.confidence)
        
    def _calculate_f1(self, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score between prediction and ground truth"""
        pred_tokens = set(prediction.strip().lower().split())
        truth_tokens = set(ground_truth.strip().lower().split())
        
        if not pred_tokens or not truth_tokens:
            return 0.0
            
        common_tokens = pred_tokens.intersection(truth_tokens)
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(truth_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics"""
        results = {}
        
        for metric_name, values in self.metrics.items():
            if values:  # Only include metrics that have been updated
                results[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                
        return results
    
    def print_metrics(self):
        """Print current metrics in a formatted way"""
        metrics = self.get_metrics()
        print("\nRAG Performance Metrics:")
        print("=" * 50)
        
        for metric_name, values in metrics.items():
            print(f"\n{metric_name.replace('_', ' ').title()}:")
            for stat_name, value in values.items():
                print(f"  {stat_name.title()}: {value:.4f}")
                
        print("\n" + "=" * 50)
    
    def calculate_scaling_metrics(self, 
                                context_lengths: List[int], 
                                performances: List[float]) -> Dict:
        """Calculate metrics related to scaling behavior"""
        # Convert to log scale for scaling law analysis
        log_lengths = np.log(context_lengths)
        log_performances = np.log(performances)
        
        # Fit linear regression to find scaling coefficient
        coefficients = np.polyfit(log_lengths, log_performances, 1)
        scaling_coefficient = coefficients[0]
        
        # Calculate R² score
        y_pred = coefficients[0] * log_lengths + coefficients[1]
        correlation = np.corrcoef(log_performances, y_pred)[0, 1]
        r2_score = correlation ** 2
        
        return {
            'scaling_coefficient': scaling_coefficient,
            'r2_score': r2_score,
            'intercept': coefficients[1]
        }