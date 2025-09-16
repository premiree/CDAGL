"""
Evaluation script for CADGL model
ICASSP 2026

Evaluates model performance under various missing modality scenarios.
"""

import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sys
sys.path.append('..')

from models.cadgl import CADGL


def evaluate_model(model, test_loader, device, scenario_name="Complete"):
    """
    Evaluate model on test set
    
    Args:
        model: Trained CADGL model
        test_loader: DataLoader for test set
        device: Computing device
        scenario_name: Name of missing modality scenario
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        # Placeholder for actual evaluation loop
        # In practice, iterate through test_loader
        print(f"Evaluating scenario: {scenario_name}")
        
        # Dummy evaluation for demonstration
        dummy_predictions = np.random.randint(0, 7, 100)
        dummy_labels = np.random.randint(0, 7, 100)
        
        all_predictions = dummy_predictions
        all_labels = dummy_labels
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'predictions': all_predictions,
        'labels': all_labels
    }


def run_all_scenarios(model, device):
    """
    Test model on all missing modality scenarios
    
    Returns results for:
    - Complete: All modalities available
    - Single missing: One modality missing at a time
    - Double missing: Two modalities missing
    """
    
    scenarios = [
        ('Complete', {'audio': 1.0, 'visual': 1.0, 'text': 1.0}),
        ('Audio Missing', {'audio': 0.0, 'visual': 1.0, 'text': 1.0}),
        ('Visual Missing', {'audio': 1.0, 'visual': 0.0, 'text': 1.0}),
        ('Text Missing', {'audio': 1.0, 'visual': 1.0, 'text': 0.0}),
        ('Audio+Visual Missing', {'audio': 0.0, 'visual': 0.0, 'text': 1.0}),
        ('Audio+Text Missing', {'audio': 0.0, 'visual': 1.0, 'text': 0.0}),
        ('Visual+Text Missing', {'audio': 1.0, 'visual': 0.0, 'text': 0.0}),
    ]
    
    results = {}
    
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    for scenario_name, missing_config in scenarios:
        # Create test loader with specific missing configuration
        # test_loader = create_test_loader(missing_config)
        
        # Evaluate
        metrics = evaluate_model(model, None, device, scenario_name)
        results[scenario_name] = metrics
        
        # Print results
        print(f"\n{scenario_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate CADGL Model')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/default.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for computation'
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Create dummy config for demo
    class Config:
        audio_dim = 74
        visual_dim = 342
        text_dim = 768
        hidden_dim = 512
        final_dim = 1536
        num_classes = 7
        dropout = 0.3
    
    config = Config()
    
    # Initialize model
    model = CADGL(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Run evaluation on all scenarios
    results = run_all_scenarios(model, device)
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\nNote: For actual evaluation, prepare test data following")
    print("the preprocessing steps described in Section 3.2 of the paper.")
    print("\nPretrained models will be available upon paper acceptance.")


if __name__ == '__main__':
    main()
