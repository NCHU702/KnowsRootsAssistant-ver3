#!/usr/bin/env python3
"""
Quick test for paper reference metadata fix
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from system_api.hierarchical_rag_system import HierarchicalRAGSystem

def test_paper_references():
    """Test that paper references display correctly"""
    
    print("="*70)
    print("Testing Paper Reference Metadata Fix")
    print("="*70)
    
    # Initialize system
    rag_system = HierarchicalRAGSystem(
        pdf_directory="./data",
        vectorstore_path="./vectorstore",
        model_name="gemma3:12b"
    )
    
    if not rag_system.is_ready():
        print("❌ RAG system not ready. Please build indices first.")
        print("\nTo build indices, run:")
        print("  python -c 'from system_api.hierarchical_rag_system import HierarchicalRAGSystem; sys = HierarchicalRAGSystem(); sys.build_indices([...])'")
        return
    
    # Test query
    query = "深度學習在醫療的應用是什麼？"
    
    print(f"\n📝 Query: {query}\n")
    print("Response (first 500 chars):")
    print("-"*70)
    
    response = rag_system.query(query)
    print(response[:500])
    
    if response.startswith("📚"):
        print("\n✅ Paper references are being displayed!")
        
        # Check if we see proper metadata
        if "Unknown" in response[:300] and "None" in response[:300]:
            print("⚠️  Warning: Still seeing 'Unknown' or 'None' in references")
            print("    This might mean the PDF metadata wasn't extracted properly")
        else:
            print("✅ Metadata looks good!")
    else:
        print("\n⚠️  No paper references found in response")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    test_paper_references()
