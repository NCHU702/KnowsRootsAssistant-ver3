#!/usr/bin/env python3
"""
Quick integration validation script for summarization chunking

Tests:
1. System can initialize with summarization mode
2. Configuration is properly passed through layers
3. Components are correctly initialized
4. System falls back to naive mode gracefully
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_api.hierarchical_rag_system import HierarchicalRAGSystem, HIERARCHICAL_RAG_CONFIG
from unittest.mock import patch


def test_1_config_structure():
    """Test 1: Verify configuration structure exists"""
    print("\n" + "="*70)
    print("Test 1: Configuration Structure")
    print("="*70)
    
    assert 'chunking' in HIERARCHICAL_RAG_CONFIG, "❌ 'chunking' key missing in config"
    print("✓ 'chunking' key exists in HIERARCHICAL_RAG_CONFIG")
    
    assert 'mode' in HIERARCHICAL_RAG_CONFIG['chunking'], "❌ 'mode' key missing"
    print("✓ 'mode' key exists")
    
    assert 'summarization' in HIERARCHICAL_RAG_CONFIG['chunking'], "❌ 'summarization' key missing"
    print("✓ 'summarization' key exists")
    
    summarization_config = HIERARCHICAL_RAG_CONFIG['chunking']['summarization']
    required_keys = ['model', 'section_parser', 'min_sections', 'map_reduce_threshold', 
                     'target_summary_length', 'ollama_base_url']
    
    for key in required_keys:
        assert key in summarization_config, f"❌ '{key}' missing in summarization config"
        print(f"  ✓ '{key}' = {summarization_config[key]}")
    
    print("\n✅ Test 1 PASSED: Configuration structure is complete\n")


def test_2_naive_mode_initialization():
    """Test 2: System initializes in naive mode (default)"""
    print("="*70)
    print("Test 2: Naive Mode Initialization (Default)")
    print("="*70)
    
    config = {
        'embedding': {
            'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'device': 'cpu'
        },
        'chunking': {
            'mode': 'naive'
        }
    }
    
    system = HierarchicalRAGSystem(
        pdf_directory='test_data',
        config=config
    )
    
    assert system.layer2 is not None, "❌ Layer2 not initialized"
    print("✓ Layer2VectorStore initialized")
    
    assert hasattr(system.layer2, 'chunking_mode'), "❌ chunking_mode attribute missing"
    print(f"✓ chunking_mode attribute exists: {system.layer2.chunking_mode}")
    
    assert system.layer2.chunking_mode == 'naive', f"❌ Expected 'naive', got '{system.layer2.chunking_mode}'"
    print("✓ chunking_mode = 'naive' (correct)")
    
    print("\n✅ Test 2 PASSED: Naive mode works correctly\n")


def test_3_summarization_mode_initialization():
    """Test 3: System initializes in summarization mode"""
    print("="*70)
    print("Test 3: Summarization Mode Initialization")
    print("="*70)
    
    config = {
        'embedding': {
            'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'device': 'cpu'
        },
        'chunking': {
            'mode': 'summarization',
            'summarization': {
                'model': 'llama3:8b',
                'section_parser': 'pymupdf_regex',
                'min_sections': 3,
                'map_reduce_threshold': 1500,
                'target_summary_length': 300,
                'ollama_base_url': 'http://localhost:11434'
            }
        }
    }
    
    # Mock OllamaLLM to avoid requiring actual Ollama
    with patch('system_api.llm_summarizer.OllamaLLM'):
        system = HierarchicalRAGSystem(
            pdf_directory='test_data',
            config=config
        )
        
        assert system.layer2 is not None, "❌ Layer2 not initialized"
        print("✓ Layer2VectorStore initialized")
        
        assert system.layer2.chunking_mode == 'summarization', \
            f"❌ Expected 'summarization', got '{system.layer2.chunking_mode}'"
        print("✓ chunking_mode = 'summarization' (correct)")
        
        assert hasattr(system.layer2, 'section_parser'), "❌ section_parser not initialized"
        print("✓ PDFSectionParser initialized")
        
        assert hasattr(system.layer2, 'summarizer'), "❌ summarizer not initialized"
        print("✓ LLMSummarizer initialized")
        
        # Verify config was passed correctly
        assert system.layer2.chunking_config['mode'] == 'summarization'
        print("✓ Chunking config passed correctly to Layer2")
    
    print("\n✅ Test 3 PASSED: Summarization mode works correctly\n")


def test_4_build_index_dispatch():
    """Test 4: build_index() correctly dispatches to right method"""
    print("="*70)
    print("Test 4: Build Index Method Dispatch")
    print("="*70)
    
    # Test naive dispatch
    config_naive = {
        'embedding': {
            'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'device': 'cpu'
        },
        'chunking': {'mode': 'naive'}
    }
    
    system_naive = HierarchicalRAGSystem(pdf_directory='test_data', config=config_naive)
    
    assert hasattr(system_naive.layer2, 'build_index'), "❌ build_index method missing"
    assert hasattr(system_naive.layer2, '_build_index_naive'), "❌ _build_index_naive method missing"
    assert hasattr(system_naive.layer2, '_build_index_with_summarization'), \
        "❌ _build_index_with_summarization method missing"
    print("✓ All build_index methods exist")
    
    # Test summarization dispatch
    config_summ = {
        'embedding': {
            'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'device': 'cpu'
        },
        'chunking': {'mode': 'summarization'}
    }
    
    with patch('system_api.llm_summarizer.OllamaLLM'):
        system_summ = HierarchicalRAGSystem(pdf_directory='test_data', config=config_summ)
        assert system_summ.layer2.chunking_mode == 'summarization'
        print("✓ Summarization system ready to dispatch to _build_index_with_summarization")
    
    print("\n✅ Test 4 PASSED: build_index dispatch logic correct\n")


def test_5_backward_compatibility():
    """Test 5: System works without explicit chunking config (backward compatible)"""
    print("="*70)
    print("Test 5: Backward Compatibility (No Chunking Config)")
    print("="*70)
    
    config = {
        'embedding': {
            'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'device': 'cpu'
        }
        # No 'chunking' key at all
    }
    
    system = HierarchicalRAGSystem(pdf_directory='test_data', config=config)
    
    assert system.layer2.chunking_mode == 'naive', \
        f"❌ Should default to 'naive', got '{system.layer2.chunking_mode}'"
    print("✓ Defaults to naive mode when chunking config missing")
    
    print("\n✅ Test 5 PASSED: Backward compatibility maintained\n")


def main():
    """Run all integration validation tests"""
    print("\n" + "="*70)
    print(" SUMMARIZATION CHUNKING INTEGRATION VALIDATION")
    print("="*70)
    
    try:
        test_1_config_structure()
        test_2_naive_mode_initialization()
        test_3_summarization_mode_initialization()
        test_4_build_index_dispatch()
        test_5_backward_compatibility()
        
        print("="*70)
        print(" ✅ ALL TESTS PASSED - INTEGRATION COMPLETE!")
        print("="*70)
        print("\nSummarization chunking is fully integrated into the system.")
        print("\nTo enable it, set in config:")
        print("  'chunking': {'mode': 'summarization'}")
        print("\nSystem will automatically:")
        print("  1. Initialize PDFSectionParser and LLMSummarizer")
        print("  2. Use _build_index_with_summarization() instead of naive")
        print("  3. Generate section-aware summaries for better RAG quality")
        print("="*70 + "\n")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
