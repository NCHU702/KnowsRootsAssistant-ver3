import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from collections import defaultdict
from langchain_ollama import OllamaLLM
from langchain_classic.prompts import PromptTemplate


@dataclass
class ResearchNode:
    """Represents a single research paper in the inheritance tree"""
    year: int
    student: str
    title: str
    area: str
    trunk: str
    note: Optional[str] = None
    predecessor: Optional['ResearchNode'] = None
    successors: List['ResearchNode'] = field(default_factory=list)
    
    def extract_methods(self) -> Set[str]:
        """Extract methods/techniques from title"""
        title_lower = self.title.lower()
        methods = set()
        
        # Deep learning models
        model_keywords = [
            'cnn', 'lstm', 'rnn', 'gan', 'cgan', 'vae', 'autoencoder',
            'transformer', 'bert', 'gpt', 'llama', 'yolo', 'unet', 'resnet',
            'dnn', 'rbf', 'svm', 'random forest', 'xgboost',
            '卷積', '循環', '生成對抗', '神經網路', '類神經', '深度學習'
        ]
        
        # Techniques
        technique_keywords = [
            'ensemble', 'transfer learning', 'federated learning', 
            'attention', 'rag', 'fine-tuning', 'pre-training',
            'data augmentation', 'reinforcement learning',
            '集成', '遷移學習', '轉移學習', '聯邦學習', '強化學習',
            '注意力機制', '資料增強', '預訓練', '微調'
        ]
        
        # Special methods
        special_keywords = [
            'gaussian', 'mdl', 'stl', 'r-tree', 'space-mdl',
            '高斯', '時空', '分解', '樹狀結構'
        ]
        
        all_keywords = model_keywords + technique_keywords + special_keywords
        
        for keyword in all_keywords:
            if keyword in title_lower:
                methods.add(keyword)
        
        return methods
    
    def get_lineage_path(self) -> List[str]:
        """Get the full lineage path"""
        path = []
        current = self
        while current:
            path.append(f"{current.year} - {current.student}")
            current = current.predecessor
        return list(reversed(path))


class ResearchInheritanceAnalyzer:
    """
    Analyzes research inheritance patterns and generates natural language reports
    using LLM to explain how topics evolve and knowledge transfers between researchers.
    """
    
    def __init__(self, json_path: str, model: str = "gemma3:12b", verbose: bool = True):
        """
        Initialize the analyzer
        
        Args:
            json_path: Path to relationships.json file
            model: Ollama model name
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self._log("Initializing Research Inheritance Analyzer...")
        
        # Load data
        with open(json_path, 'r', encoding='utf-8') as f:
            self.forest_data = json.load(f)
        
        # Initialize LLM
        self._log(f"Loading LLM model: {model}...")
        self.llm = OllamaLLM(model=model, temperature=0.7)
        
        # Build tree
        self.nodes: Dict[str, ResearchNode] = {}
        self.nodes_by_name: Dict[str, ResearchNode] = {}
        
        self._build_inheritance_tree()
        self._log(f"✓ Loaded {len(self.nodes)} research nodes\n")
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def _build_node(self, paper_data: Dict, area_name: str, trunk_name: str, 
                    predecessor: Optional[ResearchNode] = None) -> ResearchNode:
        """Build a ResearchNode from paper data"""
        node_id = f"{paper_data['year']}_{paper_data['student']}"
        
        if node_id in self.nodes:
            return self.nodes[node_id]
        
        node = ResearchNode(
            year=int(paper_data['year']),
            student=paper_data['student'],
            title=paper_data['title'],
            area=area_name,
            trunk=trunk_name,
            note=paper_data.get('note'),
            predecessor=predecessor
        )
        
        self.nodes[node_id] = node
        self.nodes_by_name[node.student] = node
        
        if predecessor:
            predecessor.successors.append(node)
        
        # Recursively build successors
        for child_data in paper_data.get('children', []):
            self._build_node(child_data, area_name, trunk_name, predecessor=node)
        
        return node
    
    def _build_inheritance_tree(self):
        """Build the complete inheritance tree"""
        for area in self.forest_data['research_forest']:
            area_name = area['area_name']
            
            for trunk in area['trunks']:
                trunk_name = trunk['trunk_name']
                
                for lineage_root in trunk['lineage']:
                    self._build_node(lineage_root, area_name, trunk_name)
    
    def _get_all_successors(self, node: ResearchNode) -> List[ResearchNode]:
        """Get all successors recursively"""
        successors = []
        for successor in node.successors:
            successors.append(successor)
            successors.extend(self._get_all_successors(successor))
        return successors
    
    def _find_researcher(self, name: str) -> Optional[ResearchNode]:
        """Find researcher by partial name match"""
        for node in self.nodes_by_name.values():
            if name.lower() in node.student.lower():
                return node
        return None
    def list_authors(
        self,
        area: Optional[str] = None,
        trunk: Optional[str] = None,
        sort: str = "alpha",          # "alpha" or "year"
        include_year: bool = False,   # if True, return ["2019 李東錡", ...]; else ["李東錡", ...]
    ) -> List[str]:
        """
        Return a unique list of available authors (students) to select from.

        Args:
            area: filter by area_name (exact match). Example: "交通數據分析"
            trunk: filter by trunk_name (exact match). Example: "人流分析與預測"
            sort: "alpha" (A→Z by name) or "year" (ascending by first appearance year)
            include_year: if True, include the earliest year next to name for display

        Returns:
            A list of author display strings.
        """
        # collect candidates
        candidates = []
        for node in self.nodes.values():
            if area and node.area != area:
                continue
            if trunk and node.trunk != trunk:
                continue
            candidates.append((node.student, node.year))

        # aggregate to earliest year per author
        first_year_by_author: Dict[str, int] = {}
        for name, yr in candidates:
            if name not in first_year_by_author or yr < first_year_by_author[name]:
                first_year_by_author[name] = yr

        # sorting
        if sort == "year":
            ordered = sorted(first_year_by_author.items(), key=lambda kv: (kv[1], kv[0]))
        else:  # alpha
            ordered = sorted(first_year_by_author.items(), key=lambda kv: kv[0])

        # formatting
        if include_year:
            return [f"{yr} {name}" for name, yr in ordered]
        else:
            return [name for name, _ in ordered]
    def list_selection_catalog(
        self,
        area: Optional[str] = None,
        trunk: Optional[str] = None,
        sort: str = "alpha",          # "alpha" or "year"
        include_year: bool = False,   # if True => "YYYY 姓名"
    ) -> Dict[str, object]:
        """
        Returns a catalog for selection UIs:
          - authors: filtered, unique, sorted display strings
          - available_areas: all areas present
          - available_trunks_by_area: {area: [trunk, ...]}
          - available_years: all years present (ints), sorted ascending

        Filters only affect the 'authors' list; the 'available_*' fields reflect the whole dataset.
        """

        # ---------- Collect globals from all nodes ----------
        areas_set: Set[str] = set()
        trunks_by_area: Dict[str, Set[str]] = defaultdict(set)
        years_set: Set[int] = set()

        for node in self.nodes.values():
            areas_set.add(node.area)
            trunks_by_area[node.area].add(node.trunk)
            years_set.add(int(node.year))

        # ---------- Collect candidates under filters ----------
        candidates: List[tuple[str, int]] = []
        for node in self.nodes.values():
            if area and node.area != area:
                continue
            if trunk and node.trunk != trunk:
                continue
            candidates.append((node.student, int(node.year)))

        # If no filter given and you want all authors, candidates already covers everyone
        # Build earliest-year per author
        first_year_by_author: Dict[str, int] = {}
        for name, yr in candidates:
            if name not in first_year_by_author or yr < first_year_by_author[name]:
                first_year_by_author[name] = yr

        # ---------- Sort authors ----------
        if sort == "year":
            ordered = sorted(first_year_by_author.items(), key=lambda kv: (kv[1], kv[0]))
        else:  # default alpha
            ordered = sorted(first_year_by_author.items(), key=lambda kv: kv[0])

        # ---------- Format authors display ----------
        if include_year:
            authors_list = [f"{yr} {name}" for name, yr in ordered]
        else:
            authors_list = [name for name, _ in ordered]

        # ---------- Build output dict ----------
        return {
            "authors": authors_list,
            "available_areas": sorted(areas_set),
            "available_trunks_by_area": {a: sorted(trunks_by_area[a]) for a in sorted(trunks_by_area)},
            "available_years": sorted(years_set),
            "filters": {
                "area": area,
                "trunk": trunk,
                "sort": sort,
                "include_year": include_year,
            },
        }

    def analyze_research_lineage(self, pioneer_researcher: str) -> str:
        """
        Generate a report about a complete research lineage
        
        Args:
            pioneer_researcher: Name of the pioneering researcher
            
        Returns:
            LLM-generated lineage report
        """
        pioneer = self._find_researcher(pioneer_researcher)
        
        if not pioneer:
            return f"找不到研究者：{pioneer_researcher}"
        
        # Make sure it's a pioneer (no predecessor)
        if pioneer.predecessor is not None:
            pioneer = pioneer.predecessor
            while pioneer.predecessor is not None:
                pioneer = pioneer.predecessor
        
        # Build complete lineage
        def build_lineage_tree(node: ResearchNode, level: int = 0) -> Dict:
            methods = node.extract_methods()
            entry = {
                "level": level,
                "generation": f"第{level + 1}代" if level > 0 else "開創",
                "researcher": node.student,
                "year": node.year,
                "title": node.title,
                "methods": list(methods)
            }
            
            if node.predecessor:
                pred_methods = node.predecessor.extract_methods()
                entry["based_on"] = node.predecessor.student
                entry["inherited_methods"] = list(methods.intersection(pred_methods))
                entry["innovations"] = list(methods - pred_methods)
            
            if node.successors:
                entry["successors"] = [build_lineage_tree(s, level + 1) for s in node.successors]
            
            return entry
        
        lineage = build_lineage_tree(pioneer)
        all_successors = self._get_all_successors(pioneer)
        
        context = {
            "lineage": lineage,
            "total_generations": max([len(s.get_lineage_path()) for s in all_successors]) if all_successors else 1,
            "total_researchers": 1 + len(all_successors),
            "year_span": f"{pioneer.year}-{max(s.year for s in all_successors)}" if all_successors else str(pioneer.year),
            "area": pioneer.area,
            "trunk": pioneer.trunk
        }
        
        # Generate report
        prompt = PromptTemplate(
            input_variables=["context"],
            template="""你是一位學術傳承研究專家。請根據以下研究工作傳承數據，用中文撰寫一份詳細的傳承分析報告。

            傳承數據：
            {context}

            請撰寫一份完整的研究工作傳承報告，包含以下部分，都用列點或一句話說 (請講的超級簡潔，除了標題要完整列出以外)：

            1. 【誰繼承誰】

            2. 【年分與方法論演變】

            3. 【可能未來工作】

            """
        )
        
        context_str = json.dumps(context, ensure_ascii=False, indent=2)
        report = self.llm.invoke(prompt.format(context=context_str))
        
        return report