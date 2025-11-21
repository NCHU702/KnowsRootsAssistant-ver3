"""
Dynamic Entity Extractor (DISABLED)

This module previously implemented query-time dynamic entity extraction.
Per the requested rollback, the dynamic extraction feature is currently
disabled. Keeping a small stub here allows the repository to remain
consistent while preventing runtime usage.
"""

import logging

logger = logging.getLogger(__name__)


class DynamicEntityExtractor:
    def __init__(self, *args, **kwargs):
        logger.warning("DynamicEntityExtractor is disabled (rollback). No operations will be performed.")

    def extract_from_retrieved_chunks(self, *args, **kwargs):
        """Return a disabled result to keep compatibility with callers."""
        return {'status': 'disabled', 'entities': [], 'relationships': [], 'chunk_count': 0, 'cached_count': 0, 'extracted_count': 0}

    def get_entity_context(self, *args, **kwargs):
        return {'entity': kwargs.get('entity_name', ''), 'chunk_ids': [], 'chunk_count': 0}
