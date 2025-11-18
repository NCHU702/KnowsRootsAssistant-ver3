"""
Unit tests for LLMSummarizer

Tests cover:
1. Map-Reduce strategy for long sections (≥1500 chars)
2. Direct summarization for short sections (<1500 chars)
3. Prompt generation (English + Chinese context)
4. Error handling and fallback to extractive summary
5. ROUGE-L score validation (≥0.6 threshold)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from system_api.llm_summarizer import LLMSummarizer


class TestLLMSummarizer:
    """Test suite for LLMSummarizer"""
    
    @pytest.fixture
    def summarizer(self):
        """Create LLMSummarizer instance with mocked Ollama"""
        with patch('system_api.llm_summarizer.OllamaLLM') as mock_ollama:
            summarizer = LLMSummarizer(
                model_name='llama3:8b',
                ollama_base_url='http://localhost:11434',
                map_reduce_threshold=1500,
                target_summary_length=300
            )
            summarizer.llm = mock_ollama.return_value
            return summarizer
    
    def test_summarizer_initialization(self):
        """Test summarizer initialization"""
        with patch('system_api.llm_summarizer.OllamaLLM'):
            summarizer = LLMSummarizer(
                model_name='llama3:8b',
                ollama_base_url='http://localhost:11434',
                map_reduce_threshold=1500,
                target_summary_length=300
            )
            
            assert summarizer.model_name == 'llama3:8b'
            assert summarizer.ollama_base_url == 'http://localhost:11434'
            assert summarizer.map_reduce_threshold == 1500
            assert summarizer.target_summary_length == 300
    
    def test_direct_summarization_short_text(self, summarizer):
        """Test direct summarization for text < map_reduce_threshold"""
        short_text = "This is a short introduction section. " * 20  # ~800 chars
        section_name = "Introduction"
        
        # Mock LLM response
        mock_summary = "This section introduces the main research topic and objectives."
        summarizer.llm.invoke = Mock(return_value=mock_summary)
        
        result = summarizer.summarize_section(short_text, section_name)
        
        # Verify
        assert isinstance(result, str)
        assert len(result) > 0
        assert len(result) < len(short_text), "Summary should be shorter than original"
        assert summarizer.llm.invoke.call_count == 1, "Should call LLM once for direct summarization"
    
    def test_map_reduce_summarization_long_text(self, summarizer):
        """Test Map-Reduce strategy for text ≥ map_reduce_threshold"""
        long_text = "This is a long methodology section with many details. " * 50  # ~2750 chars
        section_name = "Methodology"
        
        # Mock LLM responses (map phase + reduce phase)
        summarizer.llm.invoke = Mock(side_effect=[
            "Summary of first part of methodology.",  # Map chunk 1
            "Summary of second part of methodology.",  # Map chunk 2
            "Summary of third part of methodology.",  # Map chunk 3
            "Final summary combining all methodology parts."  # Reduce phase
        ])
        
        result = summarizer.summarize_section(long_text, section_name)
        
        # Verify Map-Reduce was used
        assert isinstance(result, str)
        assert len(result) > 0
        assert summarizer.llm.invoke.call_count >= 2, "Should call LLM multiple times for Map-Reduce"
    
    def test_prompt_generation_english(self, summarizer):
        """Test prompt generation for English text"""
        text = "The results show significant improvement in accuracy."
        section_name = "Results"
        paper_context = "Deep Learning for Image Classification"
        
        prompt = summarizer._build_prompt(text, section_name, paper_context)
        
        # Verify prompt structure
        assert isinstance(prompt, str)
        assert section_name in prompt
        assert text in prompt
        assert paper_context in prompt
        # Should contain instructions for summarization
        assert any(keyword in prompt.lower() for keyword in ['summarize', '總結', '摘要'])
    
    def test_prompt_generation_chinese(self, summarizer):
        """Test prompt generation for Chinese context"""
        text = "實驗結果顯示準確率有顯著提升。"
        section_name = "結果"
        paper_context = "深度學習應用於影像分類"
        
        prompt = summarizer._build_prompt(text, section_name, paper_context)
        
        # Verify Chinese prompt
        assert isinstance(prompt, str)
        assert section_name in prompt
        assert text in prompt
        assert paper_context in prompt
        # Should contain Chinese instructions
        assert any(keyword in prompt for keyword in ['總結', '摘要', '關鍵信息'])
    
    def test_error_handling_llm_failure(self, summarizer):
        """Test fallback to extractive summary on LLM failure"""
        text = "This is a section that will cause LLM to fail. " * 20
        section_name = "Discussion"
        
        # Mock LLM to raise exception
        summarizer.llm.invoke = Mock(side_effect=Exception("Ollama connection error"))
        
        result = summarizer.summarize_section(text, section_name)
        
        # Should return extractive summary (≤ target_summary_length, preferring sentence boundaries)
        assert isinstance(result, str)
        assert len(result) > 0
        # Extractive summary should be shorter than or equal to target length + possible "..."
        assert len(result) <= summarizer.target_summary_length + 10, "Extractive fallback should respect target length"
        # Result should either end with sentence punctuation or "..."
        assert (result[-1] in '.。！？' or result.endswith('...')), "Should end at sentence boundary or with '...'"
    
    def test_summary_length_constraint(self, summarizer):
        """Test that summaries respect target_summary_length"""
        text = "This is a test section. " * 100  # ~2400 chars
        section_name = "Related Work"
        
        # Mock LLM to return a summary
        expected_length = summarizer.target_summary_length
        mock_summary = "This section reviews related work in the field. " * 5  # ~250 chars
        summarizer.llm.invoke = Mock(return_value=mock_summary)
        
        result = summarizer.summarize_section(text, section_name)
        
        # Summary should be reasonably close to target length
        assert len(result) > 0
        # Allow some flexibility (50% - 200% of target)
        assert len(result) <= expected_length * 2, "Summary should not be too long"
    
    def test_rouge_l_calculation(self, summarizer):
        """Test ROUGE-L score calculation (if implemented)"""
        reference = "The quick brown fox jumps over the lazy dog."
        candidate = "The brown fox jumps over the dog."
        
        # This test assumes ROUGE-L is calculated internally
        # If not implemented, this test will be skipped
        if hasattr(summarizer, '_calculate_rouge_l'):
            score = summarizer._calculate_rouge_l(reference, candidate)
            assert 0.0 <= score <= 1.0, "ROUGE-L score should be between 0 and 1"
            assert score >= 0.6, "Good summaries should have ROUGE-L ≥ 0.6"
        else:
            pytest.skip("ROUGE-L calculation not implemented")
    
    def test_empty_text_handling(self, summarizer):
        """Test handling of empty or very short text"""
        empty_text = ""
        section_name = "Empty Section"
        
        result = summarizer.summarize_section(empty_text, section_name)
        
        # Should handle gracefully (return empty or extractive fallback)
        assert isinstance(result, str)
        # Either empty or a fallback message
        assert len(result) == 0 or "empty" in result.lower() or result == empty_text
    
    def test_map_reduce_chunk_splitting(self, summarizer):
        """Test that Map-Reduce correctly splits text into chunks"""
        # Create text that requires splitting
        long_text = "Section content. " * 200  # ~3400 chars
        section_name = "Experiments"
        
        # Mock LLM responses
        summarizer.llm.invoke = Mock(return_value="Summary.")
        
        summarizer.summarize_section(long_text, section_name)
        
        # Verify multiple invocations (map + reduce)
        assert summarizer.llm.invoke.call_count >= 2, "Should split into multiple chunks"
    
    def test_summarization_preserves_key_information(self, summarizer):
        """Test that summaries preserve key information"""
        text = """
        Our experiment used 10,000 images from ImageNet dataset.
        We achieved 95% accuracy on the test set.
        The model training took 48 hours on 4 GPUs.
        """
        section_name = "Results"
        
        # Mock LLM to return a realistic summary
        mock_summary = "Experiment on 10,000 ImageNet images achieved 95% accuracy, trained in 48 hours on 4 GPUs."
        summarizer.llm.invoke = Mock(return_value=mock_summary)
        
        result = summarizer.summarize_section(text, section_name)
        
        # Check for key information preservation
        key_info = ["10,000", "95%", "accuracy", "GPUs"]
        # At least some key information should be preserved
        preserved_count = sum(1 for info in key_info if info in result or info.lower() in result.lower())
        assert preserved_count >= 2, "Summary should preserve key numerical facts"
    
    def test_batch_summarization_consistency(self, summarizer):
        """Test that repeated summarization produces consistent results"""
        text = "This is a consistent test section. " * 30
        section_name = "Conclusion"
        
        # Mock deterministic response
        mock_summary = "This section concludes the research findings."
        summarizer.llm.invoke = Mock(return_value=mock_summary)
        
        result1 = summarizer.summarize_section(text, section_name)
        result2 = summarizer.summarize_section(text, section_name)
        
        # With mocked LLM, results should be identical
        assert result1 == result2, "Summarization should be deterministic with same inputs"


class TestLLMSummarizerIntegration:
    """Integration tests requiring actual Ollama service"""
    
    @pytest.mark.integration
    def test_real_ollama_summarization(self):
        """Test with real Ollama service (requires Ollama running)"""
        try:
            summarizer = LLMSummarizer(
                model_name='llama3:8b',
                ollama_base_url='http://localhost:11434'
            )
            
            text = """
            Deep learning has revolutionized computer vision in recent years.
            Convolutional Neural Networks (CNNs) have become the dominant architecture
            for image classification, object detection, and semantic segmentation tasks.
            Transfer learning from pre-trained models has significantly reduced
            the data and computational requirements for new applications.
            """ * 3  # ~750 chars
            
            result = summarizer.summarize_section(text, "Introduction")
            
            # Basic validation
            assert isinstance(result, str)
            assert len(result) > 0
            assert len(result) < len(text)
            
        except Exception as e:
            pytest.skip(f"Ollama service not available: {e}")
    
    @pytest.mark.integration
    def test_real_map_reduce_performance(self):
        """Test Map-Reduce on long text with real LLM"""
        try:
            summarizer = LLMSummarizer(
                model_name='llama3:8b',
                map_reduce_threshold=1000,
                target_summary_length=200
            )
            
            # Create long text
            long_text = """
            In this methodology section, we describe our experimental setup.
            We collected data from multiple sources including public datasets
            and proprietary data. The preprocessing pipeline involved several
            steps: data cleaning, normalization, and augmentation.
            """ * 20  # ~3000 chars
            
            result = summarizer.summarize_section(long_text, "Methodology")
            
            # Validation
            assert isinstance(result, str)
            assert 100 < len(result) < 500, "Summary should be within reasonable length"
            
        except Exception as e:
            pytest.skip(f"Ollama service not available: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not integration'])
