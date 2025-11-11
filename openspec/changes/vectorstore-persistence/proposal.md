# Proposal: Vector Store Persistence

## Problem Statement

Currently, the RAG system in `agent2.py` must re-embed all PDF documents every time the application starts. This process takes 6-10 minutes and includes:
- Loading 34 PDF files (~3840 document chunks)
- Embedding each chunk through Ollama's API
- Building FAISS vector store from scratch

This creates a poor developer experience and slows down the application startup significantly.

## Proposed Solution

Implement vector store persistence by:
1. Saving the FAISS index to disk after initial creation
2. Loading the saved index on subsequent startups
3. Detecting when PDFs or configurations change to trigger rebuild
4. Maintaining data integrity through hash-based validation

## Success Criteria

- ✅ First startup: Same time as current (~6-10 minutes)
- ✅ Subsequent startups: < 10 seconds (50-100x faster)
- ✅ Automatic detection of file/config changes
- ✅ Zero code changes required in `agent2.py`
- ✅ Backwards compatible with existing functionality

## Benefits

### Performance
- **Speed**: 50-100x faster startup after first run
- **Resource**: Reduces CPU/memory usage during startup
- **Scale**: Enables handling larger document collections

### Developer Experience
- Fast iteration cycles during development
- Immediate query testing without waiting
- Reduced cognitive load during debugging

### Production Readiness
- Faster service restarts
- Better reliability (less embedding API calls)
- Reduced infrastructure costs

## Technical Approach

### Core Components

1. **Hash-based Change Detection**
   - Calculate MD5 hash of all PDF files (name, mtime, size)
   - Include configuration parameters (chunk_size, chunk_overlap, model)
   - Compare with saved metadata to detect changes

2. **FAISS Index Persistence**
   - Use `FAISS.save_local()` and `FAISS.load_local()`
   - Store in configurable directory (default: `./vectorstore/`)
   - Include document metadata alongside index

3. **Metadata Management**
   - Store PDF hash, config params, timestamp
   - Use pickle for metadata serialization
   - Version tracking for future compatibility

### Files to Modify

- `system_api/rag_system.py`: Add persistence methods
  - `_get_pdf_hash()`: Calculate file fingerprint
  - `_save_vectorstore()`: Persist index to disk
  - `_load_vectorstore()`: Load saved index
  - `_initialize()`: Check for saved index before building

### New Parameters

```python
AcademicRAGSystem(
    ...,
    vectorstore_path="./vectorstore"  # New optional parameter
)
```

### Directory Structure

```
./vectorstore/
├── faiss_index.faiss      # FAISS vector index
├── faiss_index.pkl        # Document metadata
└── metadata.pkl           # PDF hash + config
```

## Implementation Plan

See [tasks.md](./tasks.md) for detailed task breakdown.

### Phase 1: Core Implementation (2-3 hours)
- Add hash calculation method
- Implement save/load functionality
- Integrate with initialization flow

### Phase 2: Testing & Validation (1-2 hours)
- Create test script for verification
- Test change detection logic
- Validate query accuracy after loading

### Phase 3: Documentation (1 hour)
- Add inline code documentation
- Create user guide
- Update README if needed

## Risk Assessment

### Low Risk
- **FAISS Compatibility**: FAISS has stable save/load API
- **Data Integrity**: Hash validation ensures accuracy
- **Backwards Compat**: Falls back to rebuild if load fails

### Mitigation Strategies
- Graceful fallback to rebuild on any load error
- Clear logging at each step
- Easy manual cache clearing (delete directory)

## Alternatives Considered

### Alternative 1: External Vector DB (Chroma, Pinecone)
- **Pros**: More features, better scaling
- **Cons**: Additional dependency, complexity, potential costs
- **Decision**: FAISS is sufficient for current scale

### Alternative 2: Incremental Updates
- **Pros**: Only embed new/changed files
- **Cons**: Complex implementation, harder to maintain
- **Decision**: Full rebuild is acceptable for this use case

### Alternative 3: No Persistence (Status Quo)
- **Pros**: Simple, no new code
- **Cons**: Poor UX, slow startup
- **Decision**: User explicitly requested improvement

## Future Enhancements

- [ ] Incremental updates for large collections
- [ ] Compression for smaller disk footprint
- [ ] Multi-version support for A/B testing
- [ ] Cloud storage integration for shared indices
- [ ] Background rebuild scheduling

## Dependencies

- **Existing**: `langchain_community.vectorstores.FAISS`
- **New**: None (uses stdlib `pickle`, `hashlib`, `pathlib`)

## Rollback Plan

If issues arise:
1. Delete `./vectorstore/` directory
2. System automatically rebuilds from scratch
3. No code changes needed to rollback

## Approval Criteria

- [ ] Code review passed
- [ ] Test script validates 50x+ speedup
- [ ] Query results match between saved/fresh indices
- [ ] Documentation complete
- [ ] User acceptance testing passed

## Timeline

- **Proposal**: 30 minutes ✅
- **Implementation**: 2-3 hours
- **Testing**: 1-2 hours
- **Documentation**: 1 hour
- **Total**: ~4-6 hours

## Open Questions

### 🔴 Critical (Need Decision Before Implementation)

1. **Upload Flow Strategy**: When a new PDF is uploaded via `/upload_paper`, how should we handle the vectorstore?
   - **Option A**: Immediate full rebuild (simple but blocks for 6-10 min)
   - **Option B**: Mark as dirty, rebuild on next restart (fast but delayed)
   - **Option C**: Incremental update (best UX but complex)
   - **Recommendation**: Start with B, plan for C in Phase 2

2. **Manual Rebuild API**: Should we expose an endpoint for forced rebuild?
   - Use case: Admin needs to rebuild without restarting service
   - Recommendation: Yes, add `/rag/rebuild` endpoint

3. **Embedding Failure Handling**: Current system has ~16% failure rate (619/3840 chunks)
   - Do we accept saving partial results (84% success)?
   - Should we track failed documents for retry?
   - Recommendation: Yes to both, add metadata tracking

### 🟡 Important (Can Decide During Implementation)

4. **Atomic Save**: Should we use atomic save operations to prevent corruption?
   - Risk: Process crashes during save
   - Solution: Save to temp directory, then atomic rename
   - Recommendation: Yes, add in Phase 1

5. **Vectorstore Versioning**: Support multiple vectorstores for different configs?
   - Use case: Testing different chunk_size or embedding models
   - Solution: Include config hash in storage path
   - Recommendation: Nice to have, defer to Phase 3

### 🟢 Low Priority (Post-MVP)

6. Should we add automatic cleanup of old vectorstore versions?
7. What's the desired behavior if PDF content changes but filename stays same?
8. Should we add disk space monitoring?

## Known Limitations

### Current System Constraints

1. **Embedding Failures**: 
   - ~16% of documents fail to embed due to Ollama limitations
   - Causes: Length limits, special characters, content formatting
   - Impact: Saved vectorstore only contains successful 84%

2. **Individual Processing**:
   - Documents processed one-by-one (not batch) due to Ollama API issues
   - 0.1s delay between documents
   - Takes 6-10 minutes for 3840 chunks

3. **PDF Upload Flow**:
   - New PDFs stored in `./data` directory
   - Currently NO automatic RAG update after upload
   - User must restart service to query new PDFs

### How This Proposal Addresses Them

- ✅ **Speeds up restart**: From 6-10 min to <10 sec
- ✅ **Preserves successful embeddings**: No need to retry successful chunks
- ⚠️ **Upload flow**: Need to add dirty marking or rebuild trigger
- ⚠️ **Failed chunks**: Will be consistently missing unless we add retry mechanism

## References

- FAISS Documentation: https://faiss.ai/
- LangChain FAISS Integration: https://python.langchain.com/docs/integrations/vectorstores/faiss
- Current implementation: `system_api/rag_system.py`

---

**Status**: 📝 Proposal - Awaiting Approval
**Author**: GitHub Copilot
**Date**: 2025-11-11
**Priority**: High (User-Requested)
