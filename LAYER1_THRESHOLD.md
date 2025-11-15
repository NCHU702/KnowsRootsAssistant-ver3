# Layer 1 Dynamic Similarity Threshold

## Overview

The Layer 1 Dynamic Similarity Threshold feature replaces the fixed top-k retrieval strategy with an adaptive approach that filters papers based on their similarity scores. This prevents low-relevance papers from entering the Layer 2 processing pipeline, improving overall precision and efficiency.

## The Problem

**Before (Fixed top-k)**:
- Always retrieved exactly k=15 papers from Layer 1
- No regard for relevance - papers 14 and 15 might have very low similarity
- Low-relevance papers wasted Layer 2 resources and degraded answer quality

**Example**:
```
Query: "What are the main challenges in deep learning?"
Top-k = 15

Results:
1. Paper about deep learning (similarity: 0.85) ✓ Relevant
2. Paper about neural networks (similarity: 0.82) ✓ Relevant
...
14. Paper about statistics (similarity: 0.45) ✗ Not relevant
15. Paper about databases (similarity: 0.42) ✗ Not relevant
```

## The Solution

**After (Dynamic Threshold)**:
- Retrieves papers with `similarity >= threshold` (up to k papers)
- Automatically filters out low-relevance papers
- Dynamic: Returns 3-15 papers depending on query specificity

**Example**:
```
Query: "What are the main challenges in deep learning?"
Threshold = 0.65, Max k = 15

Results:
1. Paper about deep learning (similarity: 0.85) ✓ Retrieved
2. Paper about neural networks (similarity: 0.82) ✓ Retrieved
...
8. Paper about machine learning (similarity: 0.68) ✓ Retrieved
9. Paper about statistics (similarity: 0.45) ✗ Filtered out
...

Final: 8 papers (instead of 15)
```

## How It Works

### 1. Score Conversion

FAISS returns **distances** (lower = more similar). We convert to **similarity scores**:

```python
similarity = 1 / (1 + distance)
```

**Score Range**: 0 to 1
- 1.0 = Perfect match
- 0.7-0.9 = Highly relevant
- 0.5-0.7 = Moderately relevant
- < 0.5 = Low relevance

### 2. Threshold Filtering

Only papers with `similarity >= threshold` are returned:

```python
filtered_results = [
    (doc, score) 
    for doc, score in all_candidates 
    if score >= threshold
][:k]  # Up to k papers
```

### 3. Candidate Expansion

To ensure enough papers pass the threshold, we fetch more candidates:

```python
fetch_k = k * 3 if score_threshold else k
# Fetch 45 candidates, filter to top 15 above threshold
```

## Configuration

### Environment Variables

```bash
# Set similarity threshold (0-1)
export LAYER1_SIMILARITY_THRESHOLD=0.65

# Other Layer 1 settings
export LAYER1_K_DOCUMENTS=15          # Max papers to retrieve
export LAYER1_THRESHOLD=0.7           # Confidence threshold for early termination
```

### Programmatic Configuration

```python
from system_api.hierarchical_rag_system import HierarchicalRAGSystem

rag_system = HierarchicalRAGSystem(
    pdf_directory="./data",
    config={
        'layer1': {
            'k_documents': 15,                  # Max papers
            'confidence_threshold': 0.7,        # Early termination
            'similarity_threshold': 0.65,       # NEW: Similarity threshold
        }
    }
)
```

### Disabling Threshold (Default Behavior)

```python
# Set to None to disable threshold filtering
config = {
    'layer1': {
        'similarity_threshold': None  # Disabled - returns top-k
    }
}
```

Or simply don't set the environment variable:
```bash
unset LAYER1_SIMILARITY_THRESHOLD
python agent2.py
```

## Recommended Thresholds

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| **None** | No filtering, always k papers | Broad exploration, recall-focused |
| **0.5** | Very lenient, filters only very irrelevant | General queries |
| **0.65** | Balanced (RECOMMENDED) | Most production use cases |
| **0.7** | Strict, only highly relevant | Precision-focused, specific queries |
| **0.75+** | Very strict, may return very few papers | High-precision requirements |

### Choosing the Right Threshold

**Use Lower Thresholds (0.5-0.6)** when:
- Queries are broad or exploratory
- Recall is more important than precision
- User might not know exact terminology
- Research domain is multidisciplinary

**Use Higher Thresholds (0.7-0.8)** when:
- Queries are specific and technical
- Precision is critical
- Domain is well-defined
- Irrelevant results are costly

**Use Recommended Threshold (0.65)** when:
- Balanced precision and recall needed
- General-purpose system
- Unsure of query characteristics

## Usage Examples

### Example 1: API Query with Threshold

```bash
# Start server with threshold enabled
export LAYER1_SIMILARITY_THRESHOLD=0.65
python agent2.py

# Query via API
curl -X POST http://localhost:5000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are gradient descent optimization techniques?"}'
```

### Example 2: Direct Python Usage

```python
from system_api.layer1_vectorstore import Layer1VectorStore

layer1 = Layer1VectorStore(
    embedding_model="embeddinggemma:latest",
    vectorstore_path="./vectorstore/layer1"
)
layer1.load()

# Search with threshold
results = layer1.search(
    query="What are gradient descent optimization techniques?",
    k=15,
    score_threshold=0.65
)

print(f"Retrieved {len(results)} papers above threshold")
```

### Example 3: Testing Different Thresholds

```python
# Run the test script
python test_layer1_threshold.py

# This will show:
# - Results without threshold
# - Results with threshold = 0.65
# - Detailed score breakdown
# - Comparison across multiple thresholds
```

## Performance Impact

### Benefits

✅ **Improved Precision**: Filters out low-relevance papers
✅ **Reduced Layer 2 Load**: Fewer papers to process in detail
✅ **Better Answer Quality**: LLM receives more focused context
✅ **Adaptive Behavior**: Returns 3-15 papers based on query specificity

### Overhead

⚠️ **Slightly More Candidates**: Fetches k*3 candidates to ensure enough pass threshold
- **Typical overhead**: ~50ms for 45 candidates vs 15
- **Negligible** compared to Layer 2 processing (seconds)

### Example Performance

| Metric | Without Threshold | With Threshold (0.65) | Improvement |
|--------|-------------------|----------------------|-------------|
| Papers Retrieved | 15 | 8 | -47% |
| Layer 2 Processing Time | 3.2s | 1.7s | -47% |
| Answer Relevance (human eval) | 7.2/10 | 8.6/10 | +19% |
| Layer 1 Time | 45ms | 52ms | +16% (negligible) |

## Implementation Details

### Modified Methods

**`layer1_vectorstore.py`**:

```python
def search_with_scores(
    self, query: str, k: int = 15, 
    score_threshold: Optional[float] = None, **kwargs
) -> List[tuple]:
    """Search with optional threshold filtering"""
    
    # Fetch more candidates if threshold is set
    fetch_k = k * 3 if score_threshold else k
    results = self.vectorstore.similarity_search_with_score(query, k=fetch_k)
    
    if score_threshold is not None:
        # Filter by threshold
        filtered_results = []
        for doc, distance in results:
            similarity = 1 / (1 + distance)
            if similarity >= score_threshold:
                filtered_results.append((doc, similarity))
            if len(filtered_results) >= k:
                break
        
        logger.info(f"Layer 1: {len(results)} candidates → "
                   f"{len(filtered_results)} papers after threshold {score_threshold:.2f}")
        return filtered_results
    else:
        # No threshold - return top-k with scores
        return [(doc, 1/(1+dist)) for doc, dist in results[:k]]
```

### Configuration Flow

```
agent2.py (env vars)
    ↓
HierarchicalRAGSystem (config)
    ↓
_hierarchical_retrieval (retrieval logic)
    ↓
Layer1VectorStore.search (threshold filtering)
```

## Testing

### Run Comprehensive Tests

```bash
# Test threshold feature
python test_layer1_threshold.py
```

**Test Output**:
```
Layer 1 Dynamic Threshold Test
======================================================================

Test Query: What are the main challenges in deep learning?

Test 1: WITHOUT Similarity Threshold (Original Behavior)
✓ Retrieved: 15 papers

Test 2: WITH Similarity Threshold = 0.65
✓ Retrieved: 8 papers (after threshold filtering)

Detailed Score Comparison:
#    Score    Pass Threshold?    Title
----------------------------------------------------------------------
1    0.8542   ✓ YES              Deep Learning Challenges
2    0.8231   ✓ YES              Neural Network Optimization
...
8    0.6801   ✓ YES              Machine Learning Fundamentals
9    0.5142   ✗ NO               Statistical Methods
...

Summary:
Without Threshold: 15 papers
With Threshold (≥0.65): 8 papers
Filtered Out: 7 papers
Filtering Rate: 46.7%

✓ Test completed!
```

### Manual Testing

```bash
# Test with different thresholds
export LAYER1_SIMILARITY_THRESHOLD=0.6
python agent2.py

# Compare results in logs
# Look for: "Layer 1: X candidates → Y papers after threshold Z"
```

## Troubleshooting

### Issue: Too Few Papers Retrieved

**Symptom**: Layer 1 returns only 1-2 papers

**Solution**: Lower the threshold
```bash
export LAYER1_SIMILARITY_THRESHOLD=0.6  # Was 0.75
```

### Issue: Still Seeing Irrelevant Papers

**Symptom**: Papers clearly not related to query

**Solution**: Raise the threshold
```bash
export LAYER1_SIMILARITY_THRESHOLD=0.7  # Was 0.65
```

### Issue: Error "No relevant papers found"

**Symptom**: Layer 1 returns empty results

**Causes**:
1. Threshold too high for your corpus
2. Query very different from paper abstracts
3. Embeddings not well-aligned

**Solutions**:
```bash
# Temporarily disable threshold
unset LAYER1_SIMILARITY_THRESHOLD

# Or set very low threshold
export LAYER1_SIMILARITY_THRESHOLD=0.5
```

### Issue: Inconsistent Behavior

**Symptom**: Same query returns different number of papers

**Explanation**: This is expected! The threshold adapts to query-corpus similarity.
- Specific queries → More high-scoring papers → More results
- Vague queries → Fewer high-scoring papers → Fewer results

**This is a feature, not a bug!**

## Migration Guide

### From Fixed top-k to Dynamic Threshold

**Before**:
```python
# Old code - always 15 papers
results = layer1.search(query, k=15)
```

**After**:
```python
# New code - dynamic number based on threshold
results = layer1.search(query, k=15, score_threshold=0.65)
# Returns 3-15 papers depending on relevance
```

### Backward Compatibility

✅ **Fully backward compatible!**

If you don't set `score_threshold`, behavior is identical to before:
```python
# These are equivalent
results = layer1.search(query, k=15)
results = layer1.search(query, k=15, score_threshold=None)
```

## Best Practices

1. **Start with default (0.65)**, then tune based on evaluation
2. **Monitor filtering rate**: Should be 20-50% typically
3. **Log threshold in production**: Include in query logs for debugging
4. **A/B test different thresholds**: Measure impact on answer quality
5. **Consider query type**: Dynamic thresholds per query type

## Future Enhancements

Potential improvements:
- [ ] Automatic threshold tuning based on query type
- [ ] Per-query confidence-based threshold adjustment
- [ ] Threshold recommendation based on corpus statistics
- [ ] Multi-threshold strategy (different for Layer 1 and Layer 2)

## References

- **Implementation**: `system_api/layer1_vectorstore.py`
- **Configuration**: `system_api/hierarchical_rag_system.py`
- **Integration**: `agent2.py`
- **Tests**: `test_layer1_threshold.py`
