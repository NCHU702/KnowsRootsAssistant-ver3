import pandas as pd
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ==================== Data Structures ====================

@dataclass
class Paper:
    """Represents a research paper with its metadata"""
    idx: int
    title: str
    author: str
    year: int
    core_themes: List[str]
    models_methods: List[str]
    research_scope: List[str]
    applications: List[str]

@dataclass
class ThemeNode:
    """Represents a node in the theme hierarchy"""
    name: str
    description: str
    parent: 'ThemeNode' = None
    children: List['ThemeNode'] = None
    papers: Set[int] = None
    embedding: np.ndarray = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.papers is None:
            self.papers = set()

# ==================== Core Theme Extraction ====================

CORE_THEMES_MAPPING = {
    "人流預測": {
        "description": "Predicting and analyzing pedestrian/crowd flow patterns",
        "aliases": ["人流預測", "人流量預測", "流量預測", "crowd flow prediction", "flow forecasting", "人群流動"],
        "parent": None
    },
    "人流異常檢測": {
        "description": "Detecting and analyzing anomalies in crowd movement",
        "aliases": ["人流異常", "異常檢測", "異常預測", "anomaly detection", "outlier detection", "找出異常", "異常分析", "exception detection"],
        "parent": "人流預測"
    },
    "時空預測": {
        "description": "Spatio-temporal predictions for crowd dynamics",
        "aliases": ["時空預測", "時空分析", "空間預測", "temporal prediction", "spatio-temporal", "人群分布", "distribution prediction"],
        "parent": "人流預測"
    },
    "傳染病防治": {
        "description": "Disease prevention and epidemic management",
        "aliases": ["傳染病防治", "傳染預測", "感染控制", "disease prevention", "epidemic management", "院內感染", "infection control"],
        "parent": None
    },
    "公共管理應用": {
        "description": "Public management and emergency response",
        "aliases": ["公共管理", "交通管理", "災害管理", "公共安全", "public management", "emergency response", "救災", "disaster management"],
        "parent": None
    },
    "商業決策": {
        "description": "Business intelligence and commercial applications",
        "aliases": ["商業決策", "商業分析", "商業應用", "行銷策略", "business intelligence", "business strategy", "商圈分析"],
        "parent": "公共管理應用"
    },
    "轉移學習": {
        "description": "Transfer learning for improved model performance",
        "aliases": ["轉移學習", "遷移學習", "transfer learning", "domain adaptation", "預訓練", "fine-tuning"],
        "parent": None
    },
    "深度學習模型": {
        "description": "Deep learning architectures and methods",
        "aliases": ["深度學習", "神經網路", "深層神經網路", "deep learning", "neural networks", "machine learning", "LSTM", "CNN", "GAN"],
        "parent": None
    },
    "數據增強": {
        "description": "Data augmentation and synthetic data generation",
        "aliases": ["數據增強", "資料擴增", "合成資料", "data augmentation", "synthetic data generation", "生成資料"],
        "parent": None
    },
    "集成學習": {
        "description": "Ensemble learning methods",
        "aliases": ["集成學習", "集成方法", "ensemble learning", "ensemble methods", "multiple models"],
        "parent": "深度學習模型"
    },
    "特徵工程": {
        "description": "Feature engineering and spatial analysis",
        "aliases": ["特徵工程", "特徵提取", "feature engineering", "spatial analysis", "聚類分析", "clustering"],
        "parent": None
    }
}

class ThemeHierarchy:
    """Manages the hierarchical classification of themes with semantic embeddings"""
    
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """Initialize hierarchy with semantic model"""
        self.model = SentenceTransformer(model_name)
        self.themes: Dict[str, ThemeNode] = {}
        self._build_hierarchy()
    
    def _build_hierarchy(self):
        """Build the theme hierarchy tree with embeddings"""
        # First pass: create all nodes
        for theme_name in CORE_THEMES_MAPPING:
            node = ThemeNode(
                name=theme_name,
                description=CORE_THEMES_MAPPING[theme_name]["description"]
            )
            self.themes[theme_name] = node
        
        # Generate embeddings for each theme based on aliases
        for theme_name, theme_info in CORE_THEMES_MAPPING.items():
            # Combine aliases and description for richer embedding
            text = " ".join(theme_info["aliases"]) + " " + theme_info["description"]
            embedding = self.model.encode(text, convert_to_numpy=True)
            self.themes[theme_name].embedding = embedding
        
        # Second pass: establish parent-child relationships
        for theme_name, theme_info in CORE_THEMES_MAPPING.items():
            parent_name = theme_info["parent"]
            if parent_name and parent_name in self.themes:
                parent_node = self.themes[parent_name]
                child_node = self.themes[theme_name]
                child_node.parent = parent_node
                parent_node.children.append(child_node)
    
    def get_hierarchy_structure(self) -> Dict:
        """Return the hierarchy as a nested dictionary"""
        def build_tree(node: ThemeNode) -> Dict:
            return {
                "name": node.name,
                "description": node.description,
                "paper_count": len(node.papers),
                "children": [build_tree(child) for child in node.children]
            }
        
        roots = [node for node in self.themes.values() if node.parent is None]
        return {
            "hierarchy": [build_tree(root) for root in roots],
            "total_themes": len(self.themes)
        }
    
    def add_papers_to_theme(self, theme_name: str, paper_idx: int):
        """Add a paper to a theme"""
        if theme_name in self.themes:
            self.themes[theme_name].papers.add(paper_idx)

# ==================== Paper Processing ====================

def parse_papers_from_df(df: pd.DataFrame) -> List[Paper]:
    """Convert pandas DataFrame to Paper objects"""
    papers = []
    
    for idx, row in df.iterrows():
        try:
            year = int(row['年份']) if pd.notna(row['年份']) else 0
            
            # Split multi-value fields by Chinese delimiter
            def split_field(field):
                if pd.isna(field):
                    return []
                return [t.strip() for t in str(field).split('、') if t.strip()]
            
            paper = Paper(
                idx=idx,
                title=row['題目'].strip() if pd.notna(row['題目']) else "",
                author=row['作者'].strip() if pd.notna(row['作者']) else "",
                year=year,
                core_themes=split_field(row['核心主題與目標']),
                models_methods=split_field(row['模型與方法']),
                research_scope=split_field(row['研究範圍與資料']),
                applications=split_field(row['應用'])
            )
            papers.append(paper)
        except Exception as e:
            print(f"Warning: Skipping row {idx} due to error: {e}")
            continue
    
    return papers

# ==================== Similarity Engine ====================

class SemanticSearchEngine:
    """Semantic similarity-based search engine using embeddings"""
    
    def __init__(self, papers: List[Paper], hierarchy: ThemeHierarchy):
        self.papers = papers
        self.hierarchy = hierarchy
        self.papers_dict = {p.idx: p for p in papers}
        self.model = hierarchy.model
        self._assign_papers_to_themes()
    
    def _assign_papers_to_themes(self):
        """Assign papers to themes using semantic similarity"""
        for paper in self.papers:
            for theme_keyword in paper.core_themes:
                # Encode the theme keyword
                kw_embedding = self.model.encode(theme_keyword, convert_to_numpy=True)
                
                # Find most similar theme
                best_theme = None
                best_sim = 0.3  # threshold
                
                for theme_name, theme_node in self.hierarchy.themes.items():
                    # Calculate cosine similarity
                    similarity = np.dot(kw_embedding, theme_node.embedding) / (
                        np.linalg.norm(kw_embedding) * np.linalg.norm(theme_node.embedding) + 1e-10
                    )
                    
                    if similarity > best_sim:
                        best_sim = similarity
                        best_theme = theme_name
                
                if best_theme:
                    self.hierarchy.add_papers_to_theme(best_theme, paper.idx)
    
    def find_best_matching_theme(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find the best matching themes for a query using semantic similarity"""
        # Encode query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Calculate similarity with all themes
        scores = []
        for theme_name, theme_node in self.hierarchy.themes.items():
            similarity = np.dot(query_embedding, theme_node.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(theme_node.embedding) + 1e-10
            )
            scores.append((theme_name, float(similarity)))
        
        # Sort by similarity and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def search_by_query(self, query: str, top_k_themes: int = 1, top_k_papers: int = 5) -> Dict:
        """Search papers based on semantic similarity to query"""
        # Find matching themes
        theme_matches = self.find_best_matching_theme(query, top_k=top_k_themes)
        
        if not theme_matches or theme_matches[0][1] < 0.3:
            return {
                "status": "no_match",
                "message": f"No semantically similar theme found for: '{query}'",
                "query": query,
                "papers": []
            }
        
        # Collect papers from matched themes and their descendants
        all_paper_indices = set()
        matched_themes_info = []
        
        for theme_name, similarity in theme_matches:
            theme_node = self.hierarchy.themes[theme_name]
            matched_themes_info.append({
                "theme": theme_name,
                "similarity": round(similarity, 3)
            })
            
            # Get papers from this theme and descendants
            def get_descendant_papers(node: ThemeNode) -> Set[int]:
                all_papers = set(node.papers)
                for child in node.children:
                    all_papers.update(get_descendant_papers(child))
                return all_papers
            
            all_paper_indices.update(get_descendant_papers(theme_node))
        
        # Sort papers by year (newest first)
        results = sorted(
            [self.papers_dict[i] for i in all_paper_indices],
            key=lambda p: p.year,
            reverse=True
        )[:top_k_papers]
        
        return {
            "status": "success",
            "query": query,
            "matched_themes": matched_themes_info,
            "paper_count": len(results),
            "papers": [
                {
                    "title": p.title,
                    "author": p.author,
                    "year": p.year,
                    "core_themes": p.core_themes,
                    "applications": p.applications
                }
                for p in results
            ]
        }
    
    def search_by_semantic_keyword(self, keyword: str, top_k: int = 10) -> Dict:
        """Search papers using semantic similarity to keyword"""
        keyword_embedding = self.model.encode(keyword, convert_to_numpy=True)
        matching_papers = []
        
        for paper in self.papers:
            score = 0
            
            # Title similarity (highest weight)
            if paper.title:
                title_emb = self.model.encode(paper.title, convert_to_numpy=True)
                title_sim = np.dot(keyword_embedding, title_emb) / (
                    np.linalg.norm(keyword_embedding) * np.linalg.norm(title_emb) + 1e-10
                )
                score += title_sim * 3
            
            # Methods similarity
            if paper.models_methods:
                methods_text = " ".join(paper.models_methods)
                methods_emb = self.model.encode(methods_text, convert_to_numpy=True)
                methods_sim = np.dot(keyword_embedding, methods_emb) / (
                    np.linalg.norm(keyword_embedding) * np.linalg.norm(methods_emb) + 1e-10
                )
                score += methods_sim * 2
            
            # Applications similarity
            if paper.applications:
                apps_text = " ".join(paper.applications)
                apps_emb = self.model.encode(apps_text, convert_to_numpy=True)
                apps_sim = np.dot(keyword_embedding, apps_emb) / (
                    np.linalg.norm(keyword_embedding) * np.linalg.norm(apps_emb) + 1e-10
                )
                score += apps_sim * 2
            
            # Core themes similarity
            if paper.core_themes:
                themes_text = " ".join(paper.core_themes)
                themes_emb = self.model.encode(themes_text, convert_to_numpy=True)
                themes_sim = np.dot(keyword_embedding, themes_emb) / (
                    np.linalg.norm(keyword_embedding) * np.linalg.norm(themes_emb) + 1e-10
                )
                score += themes_sim
            
            if score > 0:
                matching_papers.append((paper, score))
        
        matching_papers.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "keyword": keyword,
            "results_count": len(matching_papers),
            "papers": [
                {
                    "title": p.title,
                    "author": p.author,
                    "year": p.year,
                    "relevance_score": round(score, 3)
                }
                for p, score in matching_papers[:top_k]
            ]
        }

# ==================== Main SearchEngine Class ====================

class SearchEngine:
    """
    Unified search engine class that loads papers and provides search functionality
    """
    
    def __init__(
        self,
        csv_path: str,
        model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        verbose: bool = True
    ):
        """
        Initialize the SearchEngine by loading papers and setting up the semantic search system
        
        Args:
            csv_path: Path to the CSV file containing papers
            model_name: Name of the sentence transformer model to use
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self._log(f"Loading papers from {csv_path}...")
        
        # Load papers
        df = pd.read_csv(csv_path, encoding='utf-8')
        self.papers = parse_papers_from_df(df)
        self._log(f"✓ Loaded {len(self.papers)} papers")
        
        # Initialize semantic search system
        self._log("Initializing semantic search system...")
        self.hierarchy = ThemeHierarchy(model_name=model_name)
        self.engine = SemanticSearchEngine(self.papers, self.hierarchy)
        self._log(f"✓ Search engine ready with {len(self.hierarchy.themes)} themes")
    
    def _log(self, message: str):
        """Internal logging helper"""
        if self.verbose:
            print(message)
    
    def search(
        self,
        query: str | List[str],
        search_type: str = 'query',
        top_k: int = 5,
        return_json: bool = False,
        show_hierarchy: bool = False,
        show_statistics: bool = False
    ) -> Dict | str:
        """
        Search for papers using semantic similarity
        
        Args:
            query: Search query string or list of query strings
            search_type: Type of search - 'query' (theme-based) or 'keyword' (keyword-based)
            top_k: Number of results to return per query
            return_json: If True, return JSON string; if False, return dict
            show_hierarchy: If True, include hierarchy structure in results
            show_statistics: If True, include theme statistics in results
            
        Returns:
            Dictionary or JSON string containing search results
        """
        # Handle single query vs multiple queries
        queries = [query] if isinstance(query, str) else query
        
        # Perform searches
        results = {
            "search_type": search_type,
            "total_queries": len(queries),
            "results": []
        }
        
        for q in queries:
            if search_type == 'query':
                result = self.engine.search_by_query(q, top_k_themes=2, top_k_papers=top_k)
            elif search_type == 'keyword':
                result = self.engine.search_by_semantic_keyword(q, top_k=top_k)
            else:
                raise ValueError(f"Invalid search_type: {search_type}. Must be 'query' or 'keyword'")
            
            results["results"].append(result)
        
        # Add optional information
        if show_hierarchy:
            results["hierarchy"] = self.get_hierarchy()
        
        if show_statistics:
            results["statistics"] = self.get_statistics()
        
        # Return as JSON string or dict
        if return_json:
            return json.dumps(results, ensure_ascii=False, indent=2)
        else:
            return results
    
    def get_hierarchy(self) -> Dict:
        """
        Get the theme hierarchy structure
        
        Returns:
            Dictionary containing hierarchy structure
        """
        return self.hierarchy.get_hierarchy_structure()
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about papers per theme
        
        Returns:
            Dictionary with theme statistics
        """
        stats = {}
        for theme_name, theme_node in self.hierarchy.themes.items():
            if theme_node.papers:
                stats[theme_name] = len(theme_node.papers)
        
        return {
            "total_papers": len(self.papers),
            "total_themes": len(self.hierarchy.themes),
            "themes_with_papers": len(stats),
            "papers_per_theme": dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
        }
    
    def get_paper_by_index(self, idx: int) -> Optional[Dict]:
        """
        Get detailed information about a specific paper by index
        
        Args:
            idx: Paper index
            
        Returns:
            Dictionary with paper details or None if not found
        """
        paper = self.engine.papers_dict.get(idx)
        if paper:
            return {
                "idx": paper.idx,
                "title": paper.title,
                "author": paper.author,
                "year": paper.year,
                "core_themes": paper.core_themes,
                "models_methods": paper.models_methods,
                "research_scope": paper.research_scope,
                "applications": paper.applications
            }
        return None


# ==================== Demo Function ====================

def demo(csv_path: str = './paper_entries.csv'):
    """
    Run a demonstration of the SearchEngine class
    
    Args:
        csv_path: Path to the CSV file containing papers
    """
    print(f"\n{'='*80}")
    print(f"SEARCH ENGINE DEMO")
    print(f"{'='*80}\n")
    
    # Initialize SearchEngine
    engine = SearchEngine(csv_path, verbose=True)
    print()
    
    # Example 1: Single query search
    print("=" * 80)
    print("EXAMPLE 1: Single Query Search")
    print("=" * 80)
    query = "找出人群流動中的異常狀況"
    print(f"\nQuery: '{query}'")
    results = engine.search(query, search_type='query', top_k=3)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    
    # Example 2: Multiple queries at once
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multiple Queries")
    print("=" * 80)
    queries = [
        "outlier detection in crowd",
        "disease spread prevention",
        "深層神經網路建模"
    ]
    print(f"\nQueries: {queries}")
    results = engine.search(queries, search_type='query', top_k=2)
    
    for i, result in enumerate(results['results']):
        print(f"\n--- Result {i+1}: {result.get('query', '')} ---")
        if result.get('status') == 'success':
            print(f"Matched themes: {[t['theme'] for t in result['matched_themes']]}")
            print(f"Papers found: {result['paper_count']}")
        else:
            print(f"Status: {result.get('message', 'No results')}")
    
    # Example 3: Keyword search
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Keyword Search")
    print("=" * 80)
    keywords = ["LSTM", "異常檢測"]
    results = engine.search(keywords, search_type='keyword', top_k=3)
    
    for i, result in enumerate(results['results']):
        print(f"\n--- Keyword: {result.get('keyword', '')} ---")
        print(f"Results found: {result['results_count']}")
        if result['papers']:
            for j, p in enumerate(result['papers'][:2], 1):
                print(f"  {j}. {p['title'][:50]}... (Score: {p['relevance_score']})")
    
    # Example 4: Search with hierarchy and statistics
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Search with Additional Info")
    print("=" * 80)
    results = engine.search(
        "人流預測",
        search_type='query',
        top_k=2,
        show_statistics=True,
        return_json=False
    )
    print(f"\nStatistics included: {bool(results.get('statistics'))}")
    if results.get('statistics'):
        stats = results['statistics']
        print(f"Total papers: {stats['total_papers']}")
        print(f"Total themes: {stats['total_themes']}")
        print(f"Top 3 themes by paper count:")
        for theme, count in list(stats['papers_per_theme'].items())[:3]:
            print(f"  • {theme}: {count} papers")
    
    # Example 5: JSON output
    print("\n" + "=" * 80)
    print("EXAMPLE 5: JSON String Output")
    print("=" * 80)
    json_result = engine.search("epidemic", search_type='keyword', top_k=2, return_json=True)
    print(f"\nType: {type(json_result)}")
    print(f"Length: {len(json_result)} characters")
    print("\nFirst 200 characters:")
    print(json_result[:200] + "...")
    
    # Example 6: Get hierarchy separately
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Get Hierarchy")
    print("=" * 80)
    hierarchy = engine.get_hierarchy()
    print(f"\nTotal themes in hierarchy: {hierarchy['total_themes']}")
    print(f"Root nodes: {len(hierarchy['hierarchy'])}")
    
    # Example 7: Get statistics separately
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Get Statistics")
    print("=" * 80)
    stats = engine.get_statistics()
    print(f"\nTotal papers: {stats['total_papers']}")
    print(f"Themes with papers: {stats['themes_with_papers']}")
    print("\nAll themes with papers:")
    for theme, count in stats['papers_per_theme'].items():
        print(f"  • {theme}: {count} papers")


if __name__ == "__main__":
    demo()