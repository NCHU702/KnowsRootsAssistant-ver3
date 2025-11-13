"""
Query Logger Module

Logs queries and their results for analysis and debugging.
Provides statistics and insights into system usage and performance.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class QueryLogger:
    """
    Log and analyze query history
    
    Features:
    - Query logging with timestamps
    - Performance metrics
    - Layer usage statistics
    - Confidence score tracking
    - Export to JSON
    """
    
    def __init__(self, log_file: str = "./logs/query_log.jsonl"):
        """
        Initialize Query Logger
        
        Args:
            log_file: Path to log file (JSONL format)
        """
        self.log_file = log_file
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # In-memory cache for recent queries
        self._cache: List[Dict[str, Any]] = []
        self._cache_size = 100
        
        # Load existing logs
        self._load_cache()
        
        logger.info(f"Query Logger initialized: {log_file}")
    
    def log_query(
        self,
        query: str,
        result: Dict[str, Any],
        duration: Optional[float] = None
    ):
        """
        Log a query and its result
        
        Args:
            query: User query string
            result: Hierarchical retrieval result
            duration: Optional total duration
        """
        try:
            # Create log entry
            entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'query_length': len(query),
                'status': result.get('status', 'unknown'),
                'layers_used': result.get('layers_used', []),
                'terminated_at': result.get('terminated_at'),
                'final_confidence': result.get('final_confidence'),
                'timings': result.get('timings', {}),
                'duration': duration or result.get('timings', {}).get('total'),
            }
            
            # Add evaluation details if available
            if 'evaluations' in result:
                entry['evaluations'] = {}
                for layer, eval_data in result['evaluations'].items():
                    entry['evaluations'][layer] = {
                        'confidence': eval_data.get('confidence'),
                        'relevance': eval_data.get('relevance'),
                        'completeness': eval_data.get('completeness'),
                        'should_continue': eval_data.get('should_continue')
                    }
            
            # Add to cache
            self._cache.append(entry)
            if len(self._cache) > self._cache_size:
                self._cache.pop(0)
            
            # Append to file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            logger.debug(f"Query logged: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
    
    def _load_cache(self):
        """Load recent queries into cache"""
        if not os.path.exists(self.log_file):
            return
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # Load last N lines
                recent_lines = lines[-self._cache_size:] if len(lines) > self._cache_size else lines
                
                for line in recent_lines:
                    try:
                        entry = json.loads(line.strip())
                        self._cache.append(entry)
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Loaded {len(self._cache)} cached queries")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def get_recent_queries(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get N most recent queries
        
        Args:
            n: Number of queries to return
            
        Returns:
            List of query entries
        """
        return self._cache[-n:] if n <= len(self._cache) else self._cache
    
    def get_statistics(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics
        
        Args:
            limit: Optional limit on number of queries to analyze (None = all)
            
        Returns:
            Dictionary with statistics
        """
        # Load all queries if needed
        queries = self._load_all_queries() if limit is None else self._cache[-limit:]
        
        if not queries:
            return {
                'total_queries': 0,
                'message': 'No queries logged yet'
            }
        
        stats = {
            'total_queries': len(queries),
            'time_range': {
                'first': queries[0]['timestamp'] if queries else None,
                'last': queries[-1]['timestamp'] if queries else None
            }
        }
        
        # Status distribution
        status_counts = Counter(q['status'] for q in queries)
        stats['status_distribution'] = dict(status_counts)
        
        # Layer usage
        layer_usage = defaultdict(int)
        for q in queries:
            layers = q.get('layers_used', [])
            for layer in layers:
                layer_usage[layer] += 1
        stats['layer_usage'] = dict(layer_usage)
        
        # Termination points
        termination_counts = Counter(q.get('terminated_at') for q in queries if q.get('terminated_at'))
        stats['termination_distribution'] = dict(termination_counts)
        
        # Performance metrics
        durations = [q['duration'] for q in queries if q.get('duration')]
        if durations:
            stats['performance'] = {
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_duration': sum(durations)
            }
        
        # Confidence scores
        confidences = [q['final_confidence'] for q in queries if q.get('final_confidence')]
        if confidences:
            stats['confidence'] = {
                'avg_confidence': sum(confidences) / len(confidences),
                'min_confidence': min(confidences),
                'max_confidence': max(confidences)
            }
        
        # Query length statistics
        query_lengths = [q['query_length'] for q in queries]
        if query_lengths:
            stats['query_length'] = {
                'avg_length': sum(query_lengths) / len(query_lengths),
                'min_length': min(query_lengths),
                'max_length': max(query_lengths)
            }
        
        # Layer-specific timing
        layer_timings = defaultdict(list)
        for q in queries:
            timings = q.get('timings', {})
            for key, value in timings.items():
                if value is not None:
                    layer_timings[key].append(value)
        
        stats['timing_breakdown'] = {}
        for key, values in layer_timings.items():
            if values:
                stats['timing_breakdown'][key] = {
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        return stats
    
    def _load_all_queries(self) -> List[Dict[str, Any]]:
        """Load all queries from log file"""
        if not os.path.exists(self.log_file):
            return []
        
        queries = []
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        queries.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Failed to load all queries: {e}")
        
        return queries
    
    def search_queries(
        self,
        keyword: Optional[str] = None,
        status: Optional[str] = None,
        terminated_at: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search queries with filters
        
        Args:
            keyword: Search in query text
            status: Filter by status
            terminated_at: Filter by termination point
            min_confidence: Minimum confidence score
            max_confidence: Maximum confidence score
            limit: Maximum results to return
            
        Returns:
            List of matching query entries
        """
        queries = self._load_all_queries()
        results = []
        
        for q in queries:
            # Apply filters
            if keyword and keyword.lower() not in q['query'].lower():
                continue
            
            if status and q.get('status') != status:
                continue
            
            if terminated_at and q.get('terminated_at') != terminated_at:
                continue
            
            if min_confidence is not None:
                conf = q.get('final_confidence')
                if conf is None or conf < min_confidence:
                    continue
            
            if max_confidence is not None:
                conf = q.get('final_confidence')
                if conf is None or conf > max_confidence:
                    continue
            
            results.append(q)
            
            if len(results) >= limit:
                break
        
        return results
    
    def export_to_json(self, output_file: str, limit: Optional[int] = None):
        """
        Export queries to JSON file
        
        Args:
            output_file: Output file path
            limit: Optional limit on number of queries
        """
        try:
            queries = self._load_all_queries()
            if limit:
                queries = queries[-limit:]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(queries, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Exported {len(queries)} queries to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export queries: {e}")
    
    def clear_logs(self):
        """Clear all logs (use with caution)"""
        try:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            
            self._cache.clear()
            
            logger.info("Query logs cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear logs: {e}")
    
    def get_performance_summary(self) -> str:
        """
        Get human-readable performance summary
        
        Returns:
            Formatted summary string
        """
        stats = self.get_statistics()
        
        if stats['total_queries'] == 0:
            return "No queries logged yet."
        
        summary = []
        summary.append("=" * 60)
        summary.append("Query Performance Summary")
        summary.append("=" * 60)
        summary.append(f"Total Queries: {stats['total_queries']}")
        
        if 'performance' in stats:
            perf = stats['performance']
            summary.append(f"\nPerformance:")
            summary.append(f"  Average Duration: {perf['avg_duration']:.2f}s")
            summary.append(f"  Min Duration: {perf['min_duration']:.2f}s")
            summary.append(f"  Max Duration: {perf['max_duration']:.2f}s")
        
        if 'confidence' in stats:
            conf = stats['confidence']
            summary.append(f"\nConfidence Scores:")
            summary.append(f"  Average: {conf['avg_confidence']:.2f}")
            summary.append(f"  Min: {conf['min_confidence']:.2f}")
            summary.append(f"  Max: {conf['max_confidence']:.2f}")
        
        if 'termination_distribution' in stats:
            summary.append(f"\nTermination Points:")
            for point, count in stats['termination_distribution'].items():
                pct = (count / stats['total_queries']) * 100
                summary.append(f"  {point}: {count} ({pct:.1f}%)")
        
        if 'timing_breakdown' in stats:
            summary.append(f"\nTiming Breakdown:")
            for key, timing in stats['timing_breakdown'].items():
                summary.append(f"  {key}: {timing['avg']:.2f}s (avg)")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)
