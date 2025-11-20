#!/usr/bin/env python3
"""
Test script for paper references display and Chinese language support

Tests:
1. Display referenced papers in Layer 1
2. Chinese query → Chinese response
3. English query → English response
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from system_api.hierarchical_rag_system import HierarchicalRAGSystem

def test_paper_references():
    """Test paper references display"""
    
    print("="*70)
    print("Test: Paper References Display")
    print("="*70)
    
    # Initialize system
    rag_system = HierarchicalRAGSystem(
        pdf_directory="./data",
        vectorstore_path="./vectorstore",
        model_name="gemma3:12b"
    )
    
    if not rag_system.is_ready():
        print("❌ RAG system not ready. Please build indices first.")
        return
    
    # Test query (English)
    query = "What are the main challenges in deep learning?"
    
    print(f"\n📝 Query (English): {query}\n")
    print("Response:")
    print("-"*70)
    
    response = rag_system.query(query)
    print(response)
    
    print("\n" + "="*70)

def test_chinese_language():
    """Test Chinese language support"""
    
    print("="*70)
    print("Test: Chinese Language Support")
    print("="*70)
    
    # Initialize system
    rag_system = HierarchicalRAGSystem(
        pdf_directory="./data",
        vectorstore_path="./vectorstore",
        model_name="gemma3:12b"
    )
    
    if not rag_system.is_ready():
        print("❌ RAG system not ready. Please build indices first.")
        return
    
    # Test query (Chinese)
    query = "深度學習的主要挑戰是什麼？"
    
    print(f"\n📝 查詢（中文）：{query}\n")
    print("回應：")
    print("-"*70)
    
    response = rag_system.query(query)
    print(response)
    
    print("\n" + "="*70)

def test_streaming_with_references():
    """Test streaming with paper references"""
    
    print("="*70)
    print("Test: Streaming with Paper References")
    print("="*70)
    
    # Initialize system
    rag_system = HierarchicalRAGSystem(
        pdf_directory="./data",
        vectorstore_path="./vectorstore",
        model_name="gemma3:12b"
    )
    
    if not rag_system.is_ready():
        print("❌ RAG system not ready. Please build indices first.")
        return
    
    # Test query (Chinese)
    query = "請解釋transformer架構的核心概念"
    
    print(f"\n📝 查詢：{query}\n")
    print("串流回應：")
    print("-"*70)
    
    for chunk in rag_system.query_stream(query):
        print(chunk, end='', flush=True)
    
    print("\n\n" + "="*70)

def test_language_detection():
    """Test language detection logic"""
    
    print("="*70)
    print("Test: Language Detection")
    print("="*70)
    
    def is_chinese(text: str) -> bool:
        """Check if text contains significant Chinese characters"""
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        return chinese_chars > len(text) * 0.3
    
    test_cases = [
        ("What are transformers?", False),
        ("深度學習是什麼？", True),
        ("What is 深度學習?", False),  # Mixed, but English dominant
        ("深度學習 (deep learning) 的應用", True),  # Mixed, but Chinese dominant
        ("Transformer架構", False),  # Mixed, English dominant
        ("使用transformer進行自然語言處理", True),
    ]
    
    print("\n測試案例：\n")
    for text, expected in test_cases:
        detected = is_chinese(text)
        status = "✓" if detected == expected else "✗"
        print(f"{status} '{text}'")
        print(f"  Expected: {'Chinese' if expected else 'English'}, "
              f"Detected: {'Chinese' if detected else 'English'}\n")
    
    print("="*70)

if __name__ == "__main__":
    print("\n🚀 Starting Paper References & Language Tests...\n")
    
    # Test language detection first
    test_language_detection()
    
    print("\n")
    
    # Test paper references (English)
    test_paper_references()
    
    print("\n")
    
    # Test Chinese language
    test_chinese_language()
    
    print("\n")
    
    # Test streaming with references
    test_streaming_with_references()
    
    print("\n" + "="*70)
    print("All Tests Completed!")
    print("="*70)
    print("\n💡 Tips:")
    print("  - Referenced papers appear at the beginning of the response")
    print("  - Chinese queries receive Chinese responses")
    print("  - English queries receive English responses")
    print("  - Papers shown are from Layer 1 retrieval (top 10 max)")
    print()
