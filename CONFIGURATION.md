# Summarization Chunking Configuration Guide

## Overview

The summarization chunking feature (optimize-chunk-summarization) provides an intelligent alternative to naive fixed-size chunking by generating section-aware summaries of academic papers. This significantly improves information density in chunks (from ~30-40% to >80%), making better use of local LLM context windows.

## Configuration Structure

All summarization options are configured in `HIERARCHICAL_RAG_CONFIG` under the `chunking` key:

```python
HIERARCHICAL_RAG_CONFIG = {
    'chunking': {
        'mode': 'naive',  # or 'summarization'
        'summarization': {
            'model': 'llama3:8b',
            'section_parser': 'pymupdf_regex',
            'min_sections': 3,
            'map_reduce_threshold': 1500,
            'target_summary_length': 300,
            'ollama_base_url': 'http://localhost:11434',
            'store_original': False
        }
    },
    # ... other configs
}
```

## Configuration Options

### `chunking.mode` (required)

Controls which chunking strategy to use.

- **Type**: `str`
- **Options**:
  - `'naive'`: Traditional fixed-size chunking (~800 chars per chunk)
  - `'summarization'`: Section-aware summarization chunking (200-500 chars per summary)
- **Default**: `'naive'`
- **Backward Compatibility**: System defaults to naive mode if not specified

**Example**:
```python
'chunking': {
    'mode': 'summarization'  # Enable summarization mode
}
```

---

### `summarization.model` (required for summarization mode)

Specifies which Ollama model to use for generating summaries.

- **Type**: `str`
- **Recommended**: `'llama3:8b'` (good balance of quality and speed)
- **Alternatives**: `'llama3:70b'` (higher quality), `'mistral:7b'` (faster)
- **Default**: `'llama3:8b'`

**Example**:
```python
'summarization': {
    'model': 'llama3:8b'
}
```

---

### `summarization.section_parser` (optional)

Method for extracting sections from PDF files.

- **Type**: `str`
- **Options**:
  - `'pymupdf_regex'`: PyMuPDF + regex pattern matching (recommended)
  - Future: `'grobid'`, `'pdfplumber'`, etc.
- **Default**: `'pymupdf_regex'`

**How it works**:
- Uses PyMuPDF to extract text with layout information
- Applies regex patterns to detect section headers (English + Chinese)
- Recognizes common sections: Abstract, Introduction, Method, Results, Conclusion, References

**Example**:
```python
'summarization': {
    'section_parser': 'pymupdf_regex'
}
```

---

### `summarization.min_sections` (optional)

Minimum number of sections required before using section-based summarization.

- **Type**: `int`
- **Range**: `1-10`
- **Recommended**: `3`
- **Default**: `3`

**Behavior**:
- If PDF has ≥ `min_sections`, use section-aware summarization
- If PDF has < `min_sections`, fall back to paragraph-based chunking
- Prevents poor results on non-academic PDFs (e.g., slides, posters)

**Example**:
```python
'summarization': {
    'min_sections': 3  # Require at least 3 sections (e.g., Abstract, Method, Conclusion)
}
```

---

### `summarization.map_reduce_threshold` (optional)

Character count threshold for triggering Map-Reduce summarization.

- **Type**: `int`
- **Range**: `1000-5000`
- **Recommended**: `1500`
- **Default**: `1500`

**Strategy**:
- Sections **< threshold**: Direct summarization in one LLM call
- Sections **≥ threshold**: Map-Reduce strategy
  1. **Map**: Split into ~1000 char sub-chunks → summarize each (~200 chars)
  2. **Reduce**: Combine sub-summaries → final summary (~300-500 chars)

**Example**:
```python
'summarization': {
    'map_reduce_threshold': 1500  # Use Map-Reduce for sections ≥1500 chars
}
```

**Tuning**:
- **Lower** (1000): More Map-Reduce (slower, higher quality for long sections)
- **Higher** (3000): Less Map-Reduce (faster, may lose detail in long sections)

---

### `summarization.target_summary_length` (optional)

Target length for generated summaries (in characters).

- **Type**: `int`
- **Range**: `200-500`
- **Recommended**: `300`
- **Default**: `300`

**Behavior**:
- Guides LLM to generate summaries of approximately this length
- Actual summaries may vary ±20% depending on content
- Shorter sections may produce shorter summaries

**Example**:
```python
'summarization': {
    'target_summary_length': 300  # Aim for ~300 char summaries
}
```

**Trade-offs**:
- **Shorter** (200): More compressed, risk losing key details
- **Longer** (400-500): More detailed, uses more context window

---

### `summarization.ollama_base_url` (optional)

URL of the Ollama API server.

- **Type**: `str`
- **Format**: `http://host:port`
- **Default**: `'http://localhost:11434'`

**Example**:
```python
'summarization': {
    'ollama_base_url': 'http://localhost:11434'  # Local Ollama
}
```

**Remote Ollama**:
```python
'summarization': {
    'ollama_base_url': 'http://192.168.1.100:11434'  # Remote server
}
```

---

### `summarization.store_original` (optional)

Whether to store original section text alongside summaries.

- **Type**: `bool`
- **Default**: `False`

**Behavior**:
- `False`: Only store summaries (saves space, faster search)
- `True`: Store both summary and original text (useful for debugging, analysis)

**Example**:
```python
'summarization': {
    'store_original': True  # Keep original text for reference
}
```

**Storage Impact**:
- `False`: ~50 KB per paper (summaries only)
- `True`: ~500 KB per paper (summaries + originals)

---

## Complete Configuration Examples

### Example 1: Basic Summarization Mode

Enable summarization with all defaults:

```python
HIERARCHICAL_RAG_CONFIG = {
    'embedding': {
        'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'device': 'cpu'
    },
    'chunking': {
        'mode': 'summarization'  # Enable summarization
    }
}
```

### Example 2: Production Configuration

Optimized for production use:

```python
HIERARCHICAL_RAG_CONFIG = {
    'embedding': {
        'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'device': 'cuda'  # GPU acceleration
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
    },
    'vectorstore': {
        'layer1_file': 'vectorstore/layer1_index',
        'layer2_file': 'vectorstore/layer2_index',
        'jsonl_file': 'vectorstore/layer2_chunks.jsonl'
    },
    'layer1': {
        'bm25_weight': 0.3,
        'vector_weight': 0.7,
        'top_k': 5,
        'similarity_threshold': 0.3
    }
}
```

### Example 3: High-Quality Mode

For maximum summarization quality:

```python
'chunking': {
    'mode': 'summarization',
    'summarization': {
        'model': 'llama3:70b',  # Larger model
        'section_parser': 'pymupdf_regex',
        'min_sections': 2,  # More lenient
        'map_reduce_threshold': 1000,  # More granular Map-Reduce
        'target_summary_length': 400,  # Longer summaries
        'ollama_base_url': 'http://localhost:11434',
        'store_original': True  # Keep originals for review
    }
}
```

### Example 4: Fast Mode

For rapid prototyping or resource-constrained environments:

```python
'chunking': {
    'mode': 'summarization',
    'summarization': {
        'model': 'mistral:7b',  # Faster model
        'section_parser': 'pymupdf_regex',
        'min_sections': 3,
        'map_reduce_threshold': 2000,  # Less Map-Reduce
        'target_summary_length': 250,  # Shorter summaries
        'ollama_base_url': 'http://localhost:11434',
        'store_original': False
    }
}
```

---

## Usage Guide

### 1. Switch from Naive to Summarization Mode

**Before** (naive mode):
```python
rag_system = HierarchicalRAGSystem(pdf_directory='data/papers')
```

**After** (summarization mode):
```python
config = {'chunking': {'mode': 'summarization'}}
rag_system = HierarchicalRAGSystem(
    pdf_directory='data/papers',
    config=config
)
```

### 2. Build Index with Summarization

```python
# Initialize system
config = {
    'chunking': {
        'mode': 'summarization',
        'summarization': {
            'model': 'llama3:8b'
        }
    }
}

rag_system = HierarchicalRAGSystem(
    pdf_directory='data/papers',
    config=config
)

# Build indices (will use summarization)
result = rag_system.build_indices()

print(f"Papers processed: {result['papers_processed']}")
print(f"Chunks created: {result['chunks_created']}")
print(f"Duration: {result['duration']:.1f}s")
```

### 3. Check Chunk Types

After building with summarization mode, chunks will have metadata indicating their type:

```python
from system_api.layer2_document_store import JSONLDocumentStore

doc_store = JSONLDocumentStore('vectorstore/layer2_chunks.jsonl')
chunks = doc_store.get_all_chunks()

for chunk in chunks[:5]:
    print(f"Type: {chunk.metadata['chunk_type']}")
    print(f"Paper: {chunk.metadata['paper_id']}")
    if 'section_name' in chunk.metadata:
        print(f"Section: {chunk.metadata['section_name']}")
    print(f"Length: {len(chunk.page_content)} chars")
    print()
```

**Possible chunk_type values**:
- `'section_summary'`: LLM-generated summary
- `'extractive_summary'`: Fallback extractive summary (if LLM fails)
- `'naive_chunk'`: Fallback fixed-size chunk (if section parsing fails)

---

## Performance Characteristics

### Indexing Time

| Mode           | Papers/hour | Notes                              |
|----------------|-------------|------------------------------------|
| Naive          | ~200        | No LLM calls, very fast            |
| Summarization  | ~30-50      | Depends on Ollama model and hardware |

### Information Density

| Mode           | Relevant Content | Noise         |
|----------------|------------------|---------------|
| Naive          | 30-40%           | 60-70%        |
| Summarization  | 80-90%           | 10-20%        |

### Storage Size (per paper)

| Mode           | JSONL Size | FAISS Index |
|----------------|------------|-------------|
| Naive          | ~100 KB    | ~50 KB      |
| Summarization  | ~50 KB     | ~30 KB      |

---

## Troubleshooting

### Issue: Ollama connection errors

**Symptoms**:
```
WARNING  LLM error on attempt 1/3: Ollama connection error
WARNING  All LLM attempts failed, using extractive summary
```

**Solutions**:
1. Verify Ollama is running: `ollama list`
2. Check `ollama_base_url` in config
3. Test connection: `curl http://localhost:11434/api/tags`

---

### Issue: Too few sections detected

**Symptoms**:
```
WARNING  Only 2 sections found, using fallback chunking
```

**Solutions**:
1. Lower `min_sections` (e.g., from 3 to 2)
2. Check if PDF is actually an academic paper (slides/posters won't work well)
3. Inspect sections manually: `PDFSectionParser().parse(pdf_path)`

---

### Issue: Summaries too long/short

**Solutions**:
- Adjust `target_summary_length` (default 300)
- Try different model (llama3:70b produces better length control)

---

## Migration from Naive Mode

### Step 1: Backup existing indices

```bash
cp -r vectorstore/ vectorstore_backup/
```

### Step 2: Update configuration

```python
# In hierarchical_rag_system.py
HIERARCHICAL_RAG_CONFIG['chunking'] = {
    'mode': 'summarization'
}
```

### Step 3: Rebuild indices

```python
rag_system = HierarchicalRAGSystem(pdf_directory='data/papers')
rag_system.build_indices()  # Will use new summarization mode
```

### Step 4: Compare results

```python
# Query with old indices
# ... measure retrieval quality

# Query with new indices
# ... measure retrieval quality

# Compare information density, response accuracy
```

---

## Best Practices

1. **Start with defaults**: The default configuration is optimized for most use cases

2. **Monitor chunk types**: Check `chunk.metadata['chunk_type']` to see fallback rate
   - Target: >80% `section_summary`
   - Acceptable: 10-20% `extractive_summary` or `naive_chunk`

3. **Tune for your corpus**: If papers have unusual structure, adjust `min_sections`

4. **Use GPU for Ollama**: Dramatically speeds up summarization (2-3x faster)

5. **Test on sample first**: Try 5-10 papers before running on full corpus

---

## Related Documentation

- [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) - Overall system design
- [openspec/changes/optimize-chunk-summarization/design.md](./openspec/changes/optimize-chunk-summarization/design.md) - Detailed technical design
- [QUICKSTART.md](./QUICKSTART.md) - Getting started guide

---

## Support

For issues or questions:
1. Check [test_pdf_section_parser.py](./test_pdf_section_parser.py) and [test_llm_summarizer.py](./test_llm_summarizer.py) for usage examples
2. Review logs in `system_api/layer2_vectorstore.py` (search for "summarization")
3. Open an issue with:
   - Config used
   - Sample PDF (if possible)
   - Error logs
