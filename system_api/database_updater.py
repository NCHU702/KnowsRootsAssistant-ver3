"""
Database Updater Module

Updates data_categories.json and relationships.json with new paper information.
Uses LLM for research lineage classification.
Implements atomic transactions with rollback capability.
"""

import json
import os
import shutil
import threading
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import logging
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

# File paths
DATA_CATEGORIES_PATH = "./json_files/data_categories.json"
RELATIONSHIPS_PATH = "./json_files/relationships.json"
BACKUP_DIR = "./json_files/backups"

# Thread lock for concurrent update protection
_update_lock = threading.Lock()
_lock_timeout = 30  # seconds


class DatabaseUpdateError(Exception):
    """Custom exception for database update errors"""
    pass


class DatabaseUpdater:
    """Updates database JSON files with new paper information"""
    
    def __init__(self, model: str = "gemma3:12b", base_url: str = "http://localhost:11434"):
        """
        Initialize database updater
        
        Args:
            model: Ollama model name
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.llm = None
        self._initialize_llm()
        
        # Ensure backup directory exists
        os.makedirs(BACKUP_DIR, exist_ok=True)
    
    def _initialize_llm(self):
        """Initialize Ollama LLM connection"""
        try:
            self.llm = OllamaLLM(
                model=self.model,
                temperature=0.5,
                base_url=self.base_url
            )
            logger.info(f"Database updater LLM initialized with model {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise DatabaseUpdateError(f"Failed to connect to Ollama: {e}")
    
    def _backup_files(self) -> Dict[str, str]:
        """
        Create backup copies of JSON files
        
        Returns:
            Dictionary mapping original paths to backup paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backups = {}
        
        for file_path in [DATA_CATEGORIES_PATH, RELATIONSHIPS_PATH]:
            if os.path.exists(file_path):
                backup_path = os.path.join(
                    BACKUP_DIR,
                    f"{os.path.basename(file_path)}.backup_{timestamp}"
                )
                shutil.copy2(file_path, backup_path)
                backups[file_path] = backup_path
                logger.info(f"Created backup: {backup_path}")
        
        return backups
    
    def _restore_from_backup(self, backups: Dict[str, str]):
        """
        Restore files from backup
        
        Args:
            backups: Dictionary mapping original paths to backup paths
        """
        for original_path, backup_path in backups.items():
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, original_path)
                logger.info(f"Restored from backup: {original_path}")
    
    def _remove_backups(self, backups: Dict[str, str]):
        """
        Remove backup files after successful update
        
        Args:
            backups: Dictionary mapping original paths to backup paths
        """
        for backup_path in backups.values():
            if os.path.exists(backup_path):
                os.remove(backup_path)
                logger.debug(f"Removed backup: {backup_path}")
    
    def check_duplicate(self, paper_data: Dict) -> Tuple[bool, Optional[Dict]]:
        """
        Check if paper already exists in database
        
        Args:
            paper_data: Extracted paper metadata
            
        Returns:
            Tuple of (is_duplicate, existing_entry)
        """
        try:
            with open(DATA_CATEGORIES_PATH, 'r', encoding='utf-8') as f:
                categories = json.load(f)
            
            title = paper_data.get("論文標題", "").strip().lower()
            year = paper_data.get("年份", [])[0] if paper_data.get("年份") else None
            
            for entry in categories:
                existing_title = entry.get("論文標題", "").strip().lower()
                existing_year = entry.get("年份", [])[0] if entry.get("年份") else None
                
                if existing_title == title and existing_year == year:
                    logger.warning(f"Duplicate paper detected: {paper_data['論文標題']}")
                    return True, entry
            
            return False, None
        
        except Exception as e:
            logger.error(f"Error checking duplicates: {e}")
            return False, None
    
    def _create_lineage_prompt(self, paper_data: Dict, relationships_data: Dict) -> str:
        """
        Create prompt for research lineage classification
        
        Args:
            paper_data: Extracted paper metadata
            relationships_data: Current relationships structure
            
        Returns:
            Formatted prompt string
        """
        # Simplify relationships structure for prompt
        areas_summary = []
        for forest in relationships_data.get("research_forest", []):
            area_name = forest.get("area_name", "")
            trunks = [trunk.get("trunk_name", "") for trunk in forest.get("trunks", [])]
            areas_summary.append(f"- {area_name}: {', '.join(trunks)}")
        
        areas_text = "\n".join(areas_summary[:10])  # Limit to first 10 areas
        
        prompt = f"""請分析以下新論文應該歸類到哪個研究領域和主幹。論文可能是中文或英文。
Please analyze which research area and trunk the following new paper should be classified into. The paper may be in Chinese or English.

現有研究領域結構 / Existing Research Area Structure:
{areas_text}

新論文資訊 / New Paper Information:
標題 / Title: {paper_data.get('論文標題', '')}
研究目的 / Research Purpose: {', '.join(paper_data.get('研究目的', []))}
資料集 / Datasets: {', '.join(paper_data.get('資料集', []))}
建模方式 / Modeling: {', '.join(paper_data.get('建模', []))}
評估指標 / Metrics: {', '.join(paper_data.get('評估指標', []))}

請判斷此論文最適合的研究領域（area_name）和研究主幹（trunk_name），並提供理由。
Please determine the most suitable research area (area_name) and trunk (trunk_name) for this paper, and provide reasoning.

如果現有分類都不適合，建議新的area_name或trunk_name。
If existing classifications are not suitable, suggest a new area_name or trunk_name.

請返回以下格式的JSON（只返回JSON，不要其他文字）/ Please return JSON in the following format (JSON only):
{{
  "area_name": "研究領域名稱",
  "trunk_name": "研究主幹名稱",
  "is_new_area": false,
  "is_new_trunk": false,
  "reasoning": "分類理由說明（可用中英文混合說明）"
}}"""
        
        return prompt
    
    def classify_paper(self, paper_data: Dict) -> Dict:
        """
        Use LLM to classify paper into research area and trunk
        
        Args:
            paper_data: Extracted paper metadata
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Load current relationships
            with open(RELATIONSHIPS_PATH, 'r', encoding='utf-8') as f:
                relationships_data = json.load(f)
            
            # Create classification prompt
            prompt = self._create_lineage_prompt(paper_data, relationships_data)
            
            # Invoke LLM
            logger.info("Invoking LLM for research classification...")
            response = self.llm.invoke(prompt)
            
            # Parse response
            import re
            json_text = response.strip()
            
            # Extract JSON
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            elif not json_text.startswith('{'):
                json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
            
            classification = json.loads(json_text)
            
            # Validate required fields
            if "area_name" not in classification or "trunk_name" not in classification:
                raise ValueError("Missing area_name or trunk_name in classification")
            
            logger.info(f"Paper classified to: {classification['area_name']} / {classification['trunk_name']}")
            return classification
        
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Fallback to default classification
            return {
                "area_name": "未分類研究",
                "trunk_name": "待分類",
                "is_new_area": True,
                "is_new_trunk": True,
                "reasoning": f"自動分類失敗，需手動調整。錯誤: {str(e)}"
            }
    
    def update_data_categories(self, paper_data: Dict) -> bool:
        """
        Add new entry to data_categories.json
        Only includes the 8 required fields.
        
        Args:
            paper_data: Extracted paper metadata
            
        Returns:
            True if successful
        """
        try:
            # Load current data
            with open(DATA_CATEGORIES_PATH, 'r', encoding='utf-8') as f:
                categories = json.load(f)
            
            # Filter to only include the 8 required fields
            required_fields = [
                "論文標題",
                "研究目的",
                "年份",
                "資料集",
                "資料前處理",
                "建模",
                "評估指標",
                "其他"
            ]
            
            filtered_entry = {}
            for field in required_fields:
                filtered_entry[field] = paper_data.get(field, [] if field != "論文標題" else "")
            
            # Append new entry with only required fields
            categories.append(filtered_entry)
            
            # Write updated data
            with open(DATA_CATEGORIES_PATH, 'w', encoding='utf-8') as f:
                json.dump(categories, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Added paper to data_categories.json: {paper_data.get('論文標題')}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update data_categories.json: {e}")
            raise DatabaseUpdateError(f"Failed to update data_categories: {e}")
    
    def update_relationships(self, paper_data: Dict, classification: Dict) -> bool:
        """
        Add new entry to relationships.json
        
        Args:
            paper_data: Extracted paper metadata
            classification: Classification results from LLM
            
        Returns:
            True if successful
        """
        try:
            # Load current data
            with open(RELATIONSHIPS_PATH, 'r', encoding='utf-8') as f:
                relationships = json.load(f)
            
            forest = relationships.get("research_forest", [])
            
            area_name = classification["area_name"]
            trunk_name = classification["trunk_name"]
            is_new_area = classification.get("is_new_area", False)
            is_new_trunk = classification.get("is_new_trunk", False)
            
            # Create new lineage entry
            year = paper_data.get("年份", [""])[0]
            author = paper_data.get("作者", ["Unknown"])[0] if paper_data.get("作者") else "Unknown"
            title = paper_data.get("論文標題", "Untitled")
            
            new_entry = {
                "year": year,
                "student": author,
                "title": title,
                "children": []
            }
            
            # Find or create area
            area = None
            for a in forest:
                if a.get("area_name") == area_name:
                    area = a
                    break
            
            if not area or is_new_area:
                # Create new area
                area = {
                    "area_name": area_name,
                    "description": f"Research area for {area_name}",
                    "trunks": []
                }
                forest.append(area)
                logger.info(f"Created new research area: {area_name}")
            
            # Find or create trunk
            trunk = None
            for t in area.get("trunks", []):
                if t.get("trunk_name") == trunk_name:
                    trunk = t
                    break
            
            if not trunk or is_new_trunk:
                # Create new trunk
                trunk = {
                    "trunk_name": trunk_name,
                    "lineage": []
                }
                area["trunks"].append(trunk)
                logger.info(f"Created new trunk: {trunk_name}")
            
            # Add entry to lineage (at the end for now)
            trunk["lineage"].append(new_entry)
            
            # Write updated data
            with open(RELATIONSHIPS_PATH, 'w', encoding='utf-8') as f:
                json.dump(relationships, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Added paper to relationships.json: {title}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update relationships.json: {e}")
            raise DatabaseUpdateError(f"Failed to update relationships: {e}")
    
    def update_database(self, paper_data: Dict) -> Tuple[bool, str, Optional[Dict]]:
        """
        Perform atomic database update with classification
        
        Args:
            paper_data: Extracted paper metadata
            
        Returns:
            Tuple of (success, message, classification_result)
        """
        # Acquire lock with timeout
        acquired = _update_lock.acquire(timeout=_lock_timeout)
        if not acquired:
            return False, "Database is busy. Please try again later.", None
        
        backups = {}
        
        try:
            # Check for duplicates
            is_duplicate, existing = self.check_duplicate(paper_data)
            if is_duplicate:
                logger.warning("Duplicate paper detected, proceeding anyway")
                # Note: Could add user confirmation here in future
            
            # Step 1: Create backups
            backups = self._backup_files()
            
            # Step 2: Classify paper
            classification = self.classify_paper(paper_data)
            
            # Step 3: Update data_categories.json
            self.update_data_categories(paper_data)
            
            # Step 4: Update relationships.json
            self.update_relationships(paper_data, classification)
            
            # Step 5: Remove backups on success
            self._remove_backups(backups)
            
            logger.info("Database update completed successfully")
            return True, "Database updated successfully", classification
        
        except Exception as e:
            logger.error(f"Database update failed: {e}")
            
            # Rollback from backups
            if backups:
                logger.info("Rolling back database changes...")
                self._restore_from_backup(backups)
                logger.info("Rollback completed")
            
            return False, f"Database update failed: {str(e)}", None
        
        finally:
            # Release lock
            _update_lock.release()
    
    def remove_paper(self, paper_title: str) -> Tuple[bool, str]:
        """
        Remove a paper from both data_categories.json and relationships.json
        
        Args:
            paper_title: Title of the paper to remove
            
        Returns:
            Tuple of (success, message)
        """
        # Acquire lock
        if not _update_lock.acquire(timeout=_lock_timeout):
            return False, "Could not acquire lock for database removal"
        
        backups = None
        
        try:
            # Create backups
            backups = self._backup_files()
            
            # Step 1: Remove from data_categories.json
            with open(DATA_CATEGORIES_PATH, 'r', encoding='utf-8') as f:
                categories = json.load(f)
            
            original_count = len(categories)
            categories = [entry for entry in categories if entry.get('論文標題') != paper_title]
            removed_from_categories = original_count > len(categories)
            
            with open(DATA_CATEGORIES_PATH, 'w', encoding='utf-8') as f:
                json.dump(categories, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Removed paper from data_categories.json: {paper_title}")
            
            # Step 2: Remove from relationships.json
            with open(RELATIONSHIPS_PATH, 'r', encoding='utf-8') as f:
                relationships = json.load(f)
            
            removed_from_relationships = False
            for forest in relationships.get("research_forest", []):
                for trunk in forest.get("trunks", []):
                    # Remove from lineage (top-level papers)
                    original_lineage_count = len(trunk.get("lineage", []))
                    trunk["lineage"] = [
                        paper for paper in trunk.get("lineage", [])
                        if paper.get("title") != paper_title
                    ]
                    if len(trunk["lineage"]) < original_lineage_count:
                        removed_from_relationships = True
                        logger.debug(f"Removed paper from lineage in trunk: {trunk.get('trunk_name')}")
                    
                    # Also check and remove from children recursively
                    def remove_from_children(papers_list):
                        nonlocal removed_from_relationships
                        for paper in papers_list:
                            if paper.get("children"):
                                original_children_count = len(paper["children"])
                                paper["children"] = [
                                    child for child in paper["children"]
                                    if child.get("title") != paper_title
                                ]
                                if len(paper["children"]) < original_children_count:
                                    removed_from_relationships = True
                                    logger.debug(f"Removed paper from children of: {paper.get('title')}")
                                # Recursively check children's children
                                remove_from_children(paper["children"])
                    
                    remove_from_children(trunk.get("lineage", []))
            
            with open(RELATIONSHIPS_PATH, 'w', encoding='utf-8') as f:
                json.dump(relationships, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Removed paper from relationships.json: {paper_title}")
            
            if removed_from_categories or removed_from_relationships:
                return True, f"Paper '{paper_title}' removed successfully"
            else:
                return False, f"Paper '{paper_title}' not found in database"
        
        except Exception as e:
            logger.error(f"Failed to remove paper: {e}")
            
            # Rollback from backups
            if backups:
                logger.info("Rolling back database changes...")
                self._restore_from_backup(backups)
                logger.info("Rollback completed")
            
            return False, f"Failed to remove paper: {str(e)}"
        
        finally:
            # Release lock
            _update_lock.release()


def update_database_with_paper(paper_data: Dict, model: str = "gemma3:12b") -> Tuple[bool, str, Optional[Dict]]:
    """
    Convenience function to update database with new paper
    
    Args:
        paper_data: Extracted paper metadata
        model: Ollama model name
        
    Returns:
        Tuple of (success, message, classification_result)
    """
    updater = DatabaseUpdater(model=model)
    return updater.update_database(paper_data)
