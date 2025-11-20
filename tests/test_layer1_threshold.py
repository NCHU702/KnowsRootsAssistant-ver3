#!/usr/bin/env python3
"""
Test Script for Layer 1 Dynamic Similarity Threshold

This script tests the new dynamic threshold feature in Layer 1 retrieval.
It compares results with and without threshold filtering.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from system_api.hierarchical_rag_system import HierarchicalRAGSystem

def test_threshold_comparison():
    """Compare Layer 1 results with and without threshold"""
    
    print("="*70)
    print("Layer 1 Dynamic Threshold Test")
    print("="*70)
    
    # Test query
    query = "和鼻腔癌有關"
    
    print(f"\nTest Query: {query}\n")
    
    # Initialize system WITHOUT threshold
    print("\n" + "="*70)
    print("Test 1: WITHOUT Similarity Threshold (Original Behavior)")
    print("="*70)
    
    system_no_threshold = HierarchicalRAGSystem(
        pdf_directory="./data",
        vectorstore_path="./vectorstore",
        config={
            'layer1': {
                'k_documents': 15,
                'similarity_threshold': None  # No threshold
            }
        }
    )
    
    # Check if indices exist
    if not system_no_threshold.layer1.is_initialized:
        print("\n⚠️  Layer 1 index not found. Building indices first...")
        pdf_files = list(Path("./data").glob("*.pdf"))
        if not pdf_files:
            print("❌ No PDF files found in ./data directory")
            return
        system_no_threshold.build_indices([str(f) for f in pdf_files])
    
    # Search without threshold
    results_no_threshold = system_no_threshold.layer1.search(query, k=15)
    
    print(f"\n✓ Retrieved: {len(results_no_threshold)} papers")
    print("\nTop 5 papers:")
    for i, doc in enumerate(results_no_threshold[:5], 1):
        title = doc.metadata.get('title', 'Unknown')[:60]
        print(f"  {i}. {title}")
    
    # Initialize system WITH threshold
    print("\n" + "="*70)
    print("Test 2: WITH Similarity Threshold = 0.65")
    print("="*70)
    
    system_with_threshold = HierarchicalRAGSystem(
        pdf_directory="./data",
        vectorstore_path="./vectorstore",
        config={
            'layer1': {
                'k_documents': 15,
                'similarity_threshold': 0.65  # Threshold = 0.65
            }
        }
    )
    
    # Search with threshold
    results_with_threshold = system_with_threshold.layer1.search(
        query, k=15, score_threshold=0.65
    )
    
    print(f"\n✓ Retrieved: {len(results_with_threshold)} papers (after threshold filtering)")
    print("\nTop 5 papers:")
    for i, doc in enumerate(results_with_threshold[:5], 1):
        title = doc.metadata.get('title', 'Unknown')[:60]
        print(f"  {i}. {title}")
    
    # Get detailed scores for comparison
    print("\n" + "="*70)
    print("Detailed Score Comparison")
    print("="*70)
    
    results_with_scores = system_with_threshold.layer1.search_with_scores(
        query, k=15, score_threshold=None  # Get all scores
    )
    
    print(f"\nAll candidates (sorted by similarity):")
    print(f"{'#':<4} {'Score':<8} {'Pass Threshold?':<18} {'Title'}")
    print("-"*70)
    
    for i, (doc, score) in enumerate(results_with_scores[:15], 1):
        title = doc.metadata.get('title', 'Unknown')[:40]
        pass_threshold = "✓ YES" if score >= 0.65 else "✗ NO"
        print(f"{i:<4} {score:.4f}   {pass_threshold:<18} {title}")
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    filtered_count = sum(1 for _, score in results_with_scores if score >= 0.65)
    
    print(f"\nWithout Threshold: {len(results_no_threshold)} papers")
    print(f"With Threshold (≥0.65): {len(results_with_threshold)} papers")
    print(f"Filtered Out: {len(results_no_threshold) - filtered_count} papers")
    print(f"Filtering Rate: {(1 - filtered_count/len(results_no_threshold))*100:.1f}%")
    
    print("\n✓ Test completed!")


def test_different_thresholds():
    """Test various threshold values"""
    
    print("\n" + "="*70)
    print("Testing Different Threshold Values")
    print("="*70)
    
    query = "What are the main challenges in deep learning?"
    
    system = HierarchicalRAGSystem(
        pdf_directory="./data",
        vectorstore_path="./vectorstore"
    )
    
    if not system.layer1.is_initialized:
        print("⚠️  Please run test_threshold_comparison() first to build indices")
        return
    
    thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
    
    print(f"\nQuery: {query}\n")
    print(f"{'Threshold':<12} {'Papers Retrieved':<20} {'Filtering Rate'}")
    print("-"*60)
    
    # Get baseline (no threshold)
    baseline_results = system.layer1.search_with_scores(query, k=50, score_threshold=None)
    baseline_count = len(baseline_results)
    
    for threshold in thresholds:
        results = system.layer1.search_with_scores(
            query, k=50, score_threshold=threshold
        )
        count = len(results)
        filtering_rate = (1 - count/baseline_count) * 100 if baseline_count > 0 else 0
        
        print(f"{threshold:<12.2f} {count:<20} {filtering_rate:.1f}%")
    
    print("\n✓ Threshold comparison completed!")


if __name__ == "__main__":
    print("\n🚀 Starting Layer 1 Threshold Tests...\n")
    
    # Run main comparison test
    test_threshold_comparison()
    
    # Run threshold variation test
    test_different_thresholds()
    
    print("\n" + "="*70)
    print("All Tests Completed!")
    print("="*70)
    print("\nTo use in production:")
    print("  export LAYER1_SIMILARITY_THRESHOLD=0.65")
    print("  python agent2.py")
    print()
