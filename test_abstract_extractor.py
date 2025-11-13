"""
Unit tests for Abstract Extractor improvements

Tests:
1. Chinese abstract pattern extraction
2. Fallback metadata (title inclusion)
3. LLM generation with title hint
"""

import sys
import os
import unittest
from unittest.mock import Mock, MagicMock, patch

# Mock langchain_ollama before importing
sys.modules['langchain_ollama'] = MagicMock()

# Add system_api to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'system_api'))

from abstract_extractor import AbstractExtractor


class TestAbstractExtractorRegex(unittest.TestCase):
    """Test regex pattern matching for Chinese and English abstracts"""
    
    def setUp(self):
        """Set up mock LLM"""
        self.mock_llm = Mock()
        self.extractor = AbstractExtractor(self.mock_llm)
    
    def test_chinese_abstract_with_colon(self):
        """Test Chinese abstract with colon format: 摘要："""
        pdf_text = """
學術論文標題

摘要：本研究探討了深度學習在自然語言處理中的應用。我們提出了一種新的模型架構，
能夠有效處理多語言文本。實驗結果顯示，我們的方法在多個基準測試中取得了顯著的性能提升。
該方法特別適用於低資源語言的處理任務。

關鍵詞：深度學習、自然語言處理、多語言模型

1. 前言
本研究的主要目標是...
"""
        result = self.extractor._extract_with_regex(pdf_text)
        
        self.assertIsNotNone(result, "Should extract Chinese abstract")
        self.assertIn("深度學習", result)
        self.assertIn("自然語言處理", result)
        self.assertNotIn("關鍵詞", result, "Should not include keywords section")
        self.assertGreater(len(result), 50)
        print(f"✓ Chinese colon format extracted: {len(result)} chars")
    
    def test_chinese_abstract_newline_format(self):
        """Test Chinese abstract with newline format: 摘要 followed by content on next line"""
        pdf_text = """
研究論文

摘要
本文提出了一個創新的機器學習框架，用於處理複雜的數據分析任務。
該框架整合了多種先進技術，包括神經網絡和統計方法。
實驗證明該方法在準確性和效率方面都有顯著優勢。

關鍵詞
機器學習、數據分析

1. 緒論
隨著大數據時代的到來...
"""
        result = self.extractor._extract_with_regex(pdf_text)
        
        self.assertIsNotNone(result, "Should extract Chinese abstract with newline format")
        self.assertIn("機器學習", result)
        self.assertIn("數據分析", result)
        self.assertNotIn("關鍵詞", result)
        print(f"✓ Chinese newline format extracted: {len(result)} chars")
    
    def test_chinese_abstract_simplified(self):
        """Test simplified Chinese abstract format"""
        pdf_text = """
论文标题

摘要：本研究针对自然语言处理中的关键问题展开研究。我们设计了一种新型的
深度学习模型，能够有效提升文本理解能力。实验结果表明，该方法在多个
数据集上都取得了优异的表现。

关键词：自然语言处理、深度学习

1. 引言
近年来...
"""
        result = self.extractor._extract_with_regex(pdf_text)
        
        self.assertIsNotNone(result, "Should extract simplified Chinese abstract")
        self.assertIn("自然语言处理", result)
        self.assertNotIn("关键词", result, "Should not include simplified keywords")
        print(f"✓ Simplified Chinese extracted: {len(result)} chars")
    
    def test_english_abstract_standard(self):
        """Test standard English abstract format"""
        pdf_text = """
Research Paper Title

Abstract: This paper presents a novel approach to natural language processing
using deep learning techniques. We introduce a new architecture that improves
performance on multiple benchmark datasets. Our method demonstrates significant
improvements in both accuracy and computational efficiency.

Keywords: Deep Learning, Natural Language Processing

1. Introduction
In recent years...
"""
        result = self.extractor._extract_with_regex(pdf_text)
        
        self.assertIsNotNone(result, "Should extract English abstract")
        self.assertIn("natural language processing", result.lower())
        self.assertIn("deep learning", result.lower())
        self.assertNotIn("Keywords", result)
        print(f"✓ English abstract extracted: {len(result)} chars")
    
    def test_abstract_validation_pass(self):
        """Test abstract validation accepts valid abstracts"""
        valid_abstract = "This is a valid research abstract with sufficient content. " * 5
        self.assertTrue(self.extractor._validate_abstract(valid_abstract))
        print(f"✓ Valid abstract passes validation: {len(valid_abstract)} chars")
    
    def test_abstract_validation_fail_too_short(self):
        """Test abstract validation rejects too short abstracts"""
        short_abstract = "Too short"
        self.assertFalse(self.extractor._validate_abstract(short_abstract))
        print("✓ Too-short abstract rejected")
    
    def test_abstract_validation_fail_too_few_words(self):
        """Test abstract validation rejects abstracts with too few words"""
        few_words = "1234567890 " * 15  # 15 'words' but mostly numbers
        self.assertFalse(self.extractor._validate_abstract(few_words))
        print("✓ Abstract with too few words rejected")


class TestAbstractExtractorFallback(unittest.TestCase):
    """Test fallback behavior and metadata"""
    
    def setUp(self):
        """Set up mock LLM"""
        self.mock_llm = Mock()
        self.extractor = AbstractExtractor(self.mock_llm)
    
    def test_fallback_includes_title(self):
        """Test fallback abstract includes title from filename"""
        pdf_path = "/path/to/Deep_Learning_NLP_2024.pdf"
        pdf_text = "This is the beginning of a paper that has no clear abstract section. " * 10
        
        result = self.extractor._fallback_abstract(pdf_text, pdf_path)
        
        self.assertIn("Deep Learning NLP 2024", result, "Should include cleaned filename as title")
        self.assertIn("—", result, "Should have title separator")
        print(f"✓ Fallback includes title: '{result[:80]}...'")
    
    def test_fallback_sentence_boundary(self):
        """Test fallback tries to end at sentence boundary"""
        pdf_text = "First sentence. " * 30 + "Incomplete sent"
        
        result = self.extractor._fallback_abstract(pdf_text, None)
        
        self.assertTrue(result.endswith('.'), "Should end at sentence boundary when possible")
        self.assertLess(len(result), 500, "Should not exceed 500 chars")
        print(f"✓ Fallback respects sentence boundary: {len(result)} chars")
    
    def test_extract_title_from_path(self):
        """Test title extraction from PDF path"""
        test_cases = [
            ("/path/to/Machine_Learning_Survey.pdf", "Machine Learning Survey"),
            ("/docs/neural-networks-2024.pdf", "neural networks 2024"),
            ("/paper/AI.pdf", None),  # Too short (2 chars after cleaning)
            ("/test/a.pdf", None),  # Too short
        ]
        
        for pdf_path, expected in test_cases:
            result = self.extractor._extract_title_from_path(pdf_path)
            if expected:
                self.assertEqual(result, expected)
                print(f"✓ Title extracted: '{pdf_path}' → '{result}'")
            else:
                self.assertIsNone(result)
                print(f"✓ No title extracted for short filename: '{pdf_path}'")
    
    def test_full_fallback_metadata(self):
        """Test full extract() returns proper fallback metadata"""
        # Mock LLM to return None (generation fails)
        self.mock_llm.invoke.return_value = None
        
        pdf_path = "/test/Important_Research_Paper.pdf"
        pdf_text = "Some text that has no abstract markers. " * 20
        
        result = self.extractor.extract(pdf_path, pdf_text, paper_id="test_001")
        
        self.assertEqual(result['source'], 'fallback')
        self.assertEqual(result['confidence'], 0.3)
        self.assertEqual(result['method'], 'fallback')
        self.assertIn('title', result['metadata'])
        self.assertEqual(result['metadata']['title'], "Important Research Paper")
        self.assertIn('warning', result['metadata'])
        print("✓ Full fallback returns proper metadata")


class TestAbstractExtractorLLMGeneration(unittest.TestCase):
    """Test LLM generation with title hint"""
    
    def setUp(self):
        """Set up mock LLM"""
        self.mock_llm = Mock()
        self.extractor = AbstractExtractor(self.mock_llm)
    
    def test_llm_generation_includes_title_hint(self):
        """Test LLM generation includes filename as title hint"""
        pdf_path = "/papers/Transformer_Architecture_Analysis.pdf"
        pdf_text = "Introduction to transformers. " * 100
        
        # Mock LLM to return a valid abstract
        self.mock_llm.invoke.return_value = "This paper analyzes transformer architectures in detail, covering attention mechanisms and positional encodings."
        
        result = self.extractor._generate_with_llm(pdf_text, pdf_path)
        
        self.assertIsNotNone(result)
        # Check that LLM was called (we can't check the exact prompt easily without inspecting call args)
        self.mock_llm.invoke.assert_called_once()
        call_args = self.mock_llm.invoke.call_args[0][0]
        self.assertIn("Title: Transformer Architecture Analysis", call_args, "Prompt should include title hint")
        print("✓ LLM generation includes title hint in prompt")
    
    def test_llm_generation_retry_on_failure(self):
        """Test LLM generation retries on first failure"""
        pdf_text = "Test content. " * 100
        
        # First call fails, second succeeds
        self.mock_llm.invoke.side_effect = [
            Exception("Connection timeout"),
            "Valid generated abstract with sufficient content for validation."
        ]
        
        result = self.extractor._generate_with_llm(pdf_text, None)
        
        self.assertIsNotNone(result, "Should succeed on retry")
        self.assertEqual(self.mock_llm.invoke.call_count, 2, "Should retry once")
        print("✓ LLM generation retries on failure")
    
    def test_llm_generation_removes_prefixes(self):
        """Test LLM generation removes common prefixes"""
        pdf_text = "Test content. " * 100
        
        test_cases = [
            "摘要：這是一個測試摘要內容。",
            "Abstract: This is a test abstract content.",
            "本研究探討了重要問題。",
        ]
        
        for llm_output in test_cases:
            self.mock_llm.invoke.return_value = llm_output
            result = self.extractor._generate_with_llm(pdf_text, None)
            
            # Check that common prefixes are removed
            self.assertNotIn("摘要：", result)
            self.assertNotIn("Abstract:", result)
            self.assertNotIn("本研究", result[:3])  # Should not start with these
            print(f"✓ Prefix removed from: '{llm_output[:30]}...'")


def run_tests():
    """Run all tests with detailed output"""
    print("\n" + "="*70)
    print("Abstract Extractor Unit Tests")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestAbstractExtractorRegex))
    suite.addTests(loader.loadTestsFromTestCase(TestAbstractExtractorFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestAbstractExtractorLLMGeneration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
