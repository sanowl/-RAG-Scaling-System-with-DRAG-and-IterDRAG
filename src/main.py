from typing import List, Dict
import torch
from datasets import load_dataset
from .models.embedding_model import GeckoEmbeddingModel
from .retrieval.document_store import DocumentStore
from .rag.drag import DRAG
from .rag.iter_drag import IterDRAG
from .utils.data_types import RAGConfig, RAGExample, QueryResult

def load_wikipedia_documents() -> List[Dict[str, str]]:
    """Load Wikipedia documents from KILT"""
    dataset = load_dataset("kilt_wikipedia")
    documents = []
    for item in dataset['train']:
        documents.append({
            'content': item['text'],
            'title': item['title']
        })
    return documents[:100000]  # Limit for example

def load_examples(dataset_name: str) -> List[RAGExample]:
    """Load demonstration examples"""
    dataset = load_dataset(dataset_name)
    examples = []
    for item in dataset['train']:
        example = RAGExample(
            documents=[],  # Will be filled during processing
            query=item['question'],
            answer=item['answer']
        )
        examples.append(example)
    return examples[:1000]  # Limit for example

def main():
    # Initialize models and stores
    embedding_model = GeckoEmbeddingModel()
    document_store = DocumentStore(embedding_model)
    
    # Load configuration
    config = RAGConfig(
        num_documents=50,  # k in paper
        num_shots=8,      # m in paper
        max_iterations=5,  # n in paper
        max_doc_length=1024,
        max_context_length=1_000_000
    )
    
    # Initialize DRAG and IterDRAG
    drag_system = DRAG(document_store, embedding_model, config)
    iter_drag_system = IterDRAG(document_store, embedding_model, config)
    
    # Load documents and index them
    print("Loading and indexing Wikipedia documents...")
    documents = load_wikipedia_documents()
    document_store.add_documents(documents)
    
    # Load demonstration examples for different datasets
    datasets = ["hotpotqa", "musique", "2wikimultihopqa", "bamboogle"]
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        examples = load_examples(dataset_name)
        
        # Process queries with both DRAG and IterDRAG
        drag_results = []
        iter_drag_results = []
        
        # Test with different context lengths
        context_lengths = [16000, 32000, 128000, 1000000, 5000000]
        
        for length in context_lengths:
            print(f"\nTesting with context length: {length}")
            config.max_context_length = length
            
            # Process subset of examples
            test_examples = examples[:100]  # Limit for testing
            
            # DRAG processing
            print("Running DRAG...")
            for example in test_examples:
                result = drag_system.process_query(example.query, examples)
                if result.effective_context_length <= length:
                    drag_results.append(result)
            
            # IterDRAG processing
            print("Running IterDRAG...")
            for example in test_examples:
                result = iter_drag_system.process_query(example.query, examples)
                if result.effective_context_length <= length:
                    iter_drag_results.append(result)
        
        # Store results for this dataset
        all_results[dataset_name] = {
            'drag': drag_results,
            'iter_drag': iter_drag_results
        }
    
    # Analyze and print results
    for dataset_name, results in all_results.items():
        print(f"\nResults for {dataset_name}:")
        
        # Calculate metrics for DRAG
        print("\nDRAG Performance:")
        drag_performance = calculate_performance_metrics(results['drag'])
        print_metrics(drag_performance)
        
        # Calculate metrics for IterDRAG
        print("\nIterDRAG Performance:")
        iter_drag_performance = calculate_performance_metrics(results['iter_drag'])
        print_metrics(iter_drag_performance)

def calculate_performance_metrics(results: List[QueryResult]) -> Dict:
    """Calculate performance metrics for a list of results"""
    performance = {
        'avg_context_length': sum(r.effective_context_length for r in results) / len(results),
        'avg_confidence': sum(r.confidence for r in results) / len(results),
        'num_results': len(results)
    }
    
    # Calculate additional metrics for IterDRAG results
    if any(hasattr(r, 'sub_queries') for r in results):
        avg_iterations = sum(len(r.sub_queries) for r in results) / len(results)
        performance['avg_iterations'] = avg_iterations
    
    return performance

def print_metrics(metrics: Dict):
    """Print metrics in a formatted way"""
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.2f}")

if __name__ == "__main__":
    main()