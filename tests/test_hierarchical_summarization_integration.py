"""
Integration test for summarization-based chunking

Tests end-to-end flow:
1. HierarchicalRAGSystem initialization with summarization mode
2. Building indices with summarization
3. Verifying chunk metadata and content quality
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from system_api.hierarchical_rag_system import HierarchicalRAGSystem

# Test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')


class TestHierarchicalSummarizationIntegration:
    """Integration tests for summarization-based chunking"""
    
    @pytest.fixture
    def temp_vectorstore_dir(self, tmp_path):
        """Create temporary vectorstore directory"""
        vectorstore_dir = tmp_path / "vectorstore"
        vectorstore_dir.mkdir()
        return str(vectorstore_dir)
    
    @pytest.fixture
    def sample_pdf_path(self):
        """Get path to a sample PDF for testing"""
        if os.path.exists(TEST_DATA_DIR):
            pdfs = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.pdf')]
            if pdfs:
                return os.path.join(TEST_DATA_DIR, pdfs[0])
        return None
    
    def test_hierarchical_system_with_summarization_mode(self, temp_vectorstore_dir, sample_pdf_path):
        """Test HierarchicalRAGSystem can be initialized with summarization mode"""
        if not sample_pdf_path:
            pytest.skip("No sample PDF available")
        
        # Create config with summarization mode
        config = {
            'embedding': {
                'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'device': 'cpu'
            },
            'vectorstore': {
                'layer1_file': os.path.join(temp_vectorstore_dir, 'layer1_index'),
                'layer2_file': os.path.join(temp_vectorstore_dir, 'layer2_index'),
                'jsonl_file': os.path.join(temp_vectorstore_dir, 'layer2_chunks.jsonl')
            },
            'layer1': {
                'bm25_weight': 0.3,
                'vector_weight': 0.7,
                'top_k': 5,
                'similarity_threshold': 0.3
            },
            'chunking': {
                'mode': 'summarization',
                'summarization': {
                    'model': 'llama3:8b',
                    'section_parser': 'pymupdf_regex',
                    'min_sections': 3,
                    'map_reduce_threshold': 1500,
                    'target_summary_length': 300,
                    'ollama_base_url': 'http://localhost:11434',
                    'store_original': False
                }
            }
        }
        
        # Mock Ollama to avoid requiring real LLM
        with patch('system_api.llm_summarizer.OllamaLLM'):
            rag_system = HierarchicalRAGSystem(
                pdf_directory=TEST_DATA_DIR,
                config=config
            )
            
            # Verify system initialized correctly
            assert rag_system is not None
            assert rag_system.layer2 is not None
            assert rag_system.layer2.chunking_mode == 'summarization'
    
    def test_build_index_with_summarization_creates_summaries(self, temp_vectorstore_dir, sample_pdf_path):
        """Test that building index with summarization creates proper section summaries"""
        if not sample_pdf_path:
            pytest.skip("No sample PDF available")
        
        config = {
            'embedding': {
                'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'device': 'cpu'
            },
            'vectorstore': {
                'layer1_file': os.path.join(temp_vectorstore_dir, 'layer1_index'),
                'layer2_file': os.path.join(temp_vectorstore_dir, 'layer2_index'),
                'jsonl_file': os.path.join(temp_vectorstore_dir, 'layer2_chunks.jsonl')
            },
            'layer1': {
                'bm25_weight': 0.3,
                'vector_weight': 0.7,
                'top_k': 5,
                'similarity_threshold': 0.3
            },
            'chunking': {
                'mode': 'summarization',
                'summarization': {
                    'model': 'llama3:8b',
                    'section_parser': 'pymupdf_regex',
                    'min_sections': 3,
                    'map_reduce_threshold': 1500,
                    'target_summary_length': 300,
                    'ollama_base_url': 'http://localhost:11434',
                    'store_original': False
                }
            }
        }
        
        # Mock Ollama LLM to return predictable summaries
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "這是一個測試摘要，長度約為300字符。" * 5  # ~150 chars
        
        with patch('system_api.llm_summarizer.OllamaLLM', return_value=mock_llm):
            rag_system = HierarchicalRAGSystem(
                pdf_directory=TEST_DATA_DIR,
                config=config
            )
            
            # Build indices with single PDF
            result = rag_system.build_indices(pdf_paths=[sample_pdf_path])
            
            # Verify build succeeded
            assert result['status'] == 'success'
            assert result['chunks_created'] > 0
            
            # Verify chunks were created
            assert os.path.exists(config['vectorstore']['jsonl_file'])
            
            # Read and verify chunk metadata
            from system_api.layer2_document_store import JSONLDocumentStore
            doc_store = JSONLDocumentStore(config['vectorstore']['jsonl_file'])
            
            # Get all chunks
            paper_id = os.path.splitext(os.path.basename(sample_pdf_path))[0]
            chunks = doc_store.get_chunks_by_paper(paper_id)
            
            assert len(chunks) > 0, "Should have created chunks"
            
            # Verify chunk types and metadata
            for chunk in chunks:
                assert 'chunk_type' in chunk.metadata
                chunk_type = chunk.metadata['chunk_type']
                
                # Should be one of the expected types
                assert chunk_type in ['section_summary', 'extractive_summary', 'naive_chunk']
                
                # Should have paper_id
                assert chunk.metadata.get('paper_id') == paper_id
                
                # Section summaries should have section metadata
                if chunk_type == 'section_summary':
                    assert 'section_name' in chunk.metadata
                    assert 'original_length' in chunk.metadata
                    assert 'summary_length' in chunk.metadata
    
    def test_fallback_to_naive_mode_on_error(self, temp_vectorstore_dir, sample_pdf_path):
        """Test system falls back to naive chunking when summarization fails"""
        if not sample_pdf_path:
            pytest.skip("No sample PDF available")
        
        config = {
            'embedding': {
                'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'device': 'cpu'
            },
            'vectorstore': {
                'layer1_file': os.path.join(temp_vectorstore_dir, 'layer1_index'),
                'layer2_file': os.path.join(temp_vectorstore_dir, 'layer2_index'),
                'jsonl_file': os.path.join(temp_vectorstore_dir, 'layer2_chunks.jsonl')
            },
            'layer1': {
                'bm25_weight': 0.3,
                'vector_weight': 0.7,
                'top_k': 5,
                'similarity_threshold': 0.3
            },
            'chunking': {
                'mode': 'summarization',
                'summarization': {
                    'model': 'llama3:8b',
                    'section_parser': 'pymupdf_regex',
                    'min_sections': 3,
                    'map_reduce_threshold': 1500,
                    'target_summary_length': 300,
                    'ollama_base_url': 'http://localhost:11434',
                    'store_original': False
                }
            }
        }
        
        # Mock section parser to raise an error
        with patch('system_api.pdf_section_parser.PDFSectionParser') as mock_parser:
            mock_parser_instance = MagicMock()
            mock_parser_instance.parse.side_effect = Exception("PDF parsing failed")
            mock_parser.return_value = mock_parser_instance
            
            with patch('system_api.llm_summarizer.OllamaLLM'):
                rag_system = HierarchicalRAGSystem(
                    pdf_directory=TEST_DATA_DIR,
                    config=config
                )
                
                # Build indices should still succeed with fallback
                result = rag_system.build_indices(pdf_paths=[sample_pdf_path])
                
                # Should succeed despite errors (fallback to naive)
                assert result['status'] == 'success'
                assert result['chunks_created'] > 0
                
                # Verify chunks were created (should be naive_chunk type)
                from system_api.layer2_document_store import JSONLDocumentStore
                doc_store = JSONLDocumentStore(config['vectorstore']['jsonl_file'])
                
                paper_id = os.path.splitext(os.path.basename(sample_pdf_path))[0]
                chunks = doc_store.get_chunks_by_paper(paper_id)
                
                # All chunks should be naive_chunk type (fallback)
                assert all(c.metadata.get('chunk_type') == 'naive_chunk' for c in chunks)
    
    def test_naive_mode_still_works(self, temp_vectorstore_dir, sample_pdf_path):
        """Test that naive mode still works as before (backward compatibility)"""
        if not sample_pdf_path:
            pytest.skip("No sample PDF available")
        
        config = {
            'embedding': {
                'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'device': 'cpu'
            },
            'vectorstore': {
                'layer1_file': os.path.join(temp_vectorstore_dir, 'layer1_index'),
                'layer2_file': os.path.join(temp_vectorstore_dir, 'layer2_index'),
                'jsonl_file': os.path.join(temp_vectorstore_dir, 'layer2_chunks.jsonl')
            },
            'layer1': {
                'bm25_weight': 0.3,
                'vector_weight': 0.7,
                'top_k': 5,
                'similarity_threshold': 0.3
            },
            'chunking': {
                'mode': 'naive'  # Use naive mode explicitly
            }
        }
        
        rag_system = HierarchicalRAGSystem(
            pdf_directory=TEST_DATA_DIR,
            config=config
        )
        
        # Verify naive mode is set
        assert rag_system.layer2.chunking_mode == 'naive'
        
        # Build indices
        result = rag_system.build_indices(pdf_paths=[sample_pdf_path])
        
        # Should succeed
        assert result['status'] == 'success'
        assert result['chunks_created'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
