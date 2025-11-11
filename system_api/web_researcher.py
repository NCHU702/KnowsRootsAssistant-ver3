"""
WebSearcher Class - Academic Paper Search using LangChain
Accepts an external LLM and provides paper search functionality
"""

import re
from langchain_community.tools import DuckDuckGoSearchResults

# Regex patterns for parsing LLM output
DATE_RE = re.compile(r'([A-Z][a-z]{2} \d{1,2}, \d{4})\s*·')
LINK_RE = re.compile(r'https?://\S+')
AUTHOR_RE = re.compile(r'(?:by|author\s*=\s*[{(])\s*([\w\.\-\,\s&]+?)(?:[)}]|,|\.)', re.I)
PREFIX_RE = re.compile(r'(title|link|snippet):', re.I)

def parse_llm_organized_papers(raw: str) -> str:
    """
    Parse LLM-organized search results into a standardized markdown format.
    
    Args:
        raw: Raw string from LLM with organized paper data
        
    Returns:
        Markdown-formatted string with paper information
    """
    text = re.sub(r'\s+', ' ', raw.strip())
    formatted = ""
    idx = 1
    
    # Assume LLM returns markdown with sections like "## Title\n- Author: ...\n- Link: ...\n- Date: ..."
    for block in text.split('\n## ')[1:]:  # Skip initial header, split by section
        try:
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if not lines:
                continue
            
            title = lines[0].strip('# ').strip()  # First line is title
            author = "Unknown"
            link = ""
            dates = "Unknown Date"
            
            for line in lines[1:]:
                if line.startswith('- Author:') or line.startswith('- Authors:'):
                    author = line.replace('- Author:', '').replace('- Authors:', '').strip()
                elif line.startswith('- Link:'):
                    link_match = LINK_RE.search(line.replace('- Link:', '').strip())
                    link = link_match.group(0) if link_match else line.replace('- Link:', '').strip()
                elif line.startswith('- Date:'):
                    dates = line.replace('- Date:', '').strip()
            
            formatted += f"# {idx}. {title}\n"
            formatted += f"- **Author:** {author}\n"
            formatted += f"- **Dates:** {dates}\n"
            formatted += f"- **Link:** {link}\n"
            formatted += "---\n\n"
            idx += 1
        except (IndexError, AttributeError):
            continue  # Skip malformed blocks
    
    return formatted if formatted else raw  # Return raw if no valid blocks

class WebSearcher:
    """
    A class for searching academic papers using DuckDuckGo search and LLM organization.
    
    Attributes:
        llm: The language model instance for organizing results
    """
    
    def __init__(self, llm):
        """
        Initialize the WebSearcher with an LLM.
        
        Args:
            llm: A LangChain-compatible LLM instance (e.g., OllamaLLM)
        """
        self.llm = llm
    
    def _organize_with_llm(self, raw_results: str) -> str:
        """
        Use LLM to organize raw search results into a structured markdown format.
        
        Args:
            raw_results: Raw string from DuckDuckGo search
            
        Returns:
            Organized string from LLM (markdown format)
        """
        if not self.llm or not raw_results or raw_results == "No good DuckDuckGo Search Result was found":
            return raw_results
        
        # LLM prompt for organizing papers
        prompt = f"""
        You are an academic research assistant. Please organize the following search results into a clean, structured markdown format. 
        Extract each paper's title, author(s) (if available), publication date (if available), and link. 
        Format each paper as a separate section using the following template:

        ## TITLE OF THE PAPER
        - Author: [author name(s)]
        - Link: [full URL]
        - Date: [publication date, e.g., Dec 20, 2018]

        Focus on academic papers only. Ignore non-paper results. If information is missing, use "Unknown".
        If multiple papers are in one snippet, separate them clearly.

        Raw search results:
        {raw_results}

        Output ONLY the organized markdown sections, no additional text.
        """
        
        try:
            # Call LLM to organize results
            organized_response = self.llm.invoke(prompt)
            return organized_response
        except Exception as e:
            # logger.error(f"LLM organization failed: {e}")
            return raw_results  # Fallback to raw results if LLM fails

    def _search_papers(self, query: str) -> str:
        """
        Internal method to search for academic papers and organize with LLM.
        
        Args:
            query: Search query string
            
        Returns:
            Formatted string with paper titles and links
        """
        # Enhance query with academic sources
        enhanced_query = (
            f"{query} site:arxiv.org OR site:scholar.google.com OR "
            f"site:pubmed.ncbi.nlm.nih.gov OR filetype:pdf research paper"
        )
        
        search = DuckDuckGoSearchResults(num_results=10)
        results = search.run(enhanced_query)
        
        if not results or results == "No good DuckDuckGo Search Result was found":
            return "No papers found. Try refining your search query."
        
        # Organize with LLM
        organized_results = self._organize_with_llm(results)
        
        # Parse organized results into standardized format
        return parse_llm_organized_papers(organized_results)
    
    def search(self, user_query: str) -> str:
        """
        Main public method to search for papers based on user query.
        
        Args:
            user_query: User's research question or topic
            
        Returns:
            Formatted response with paper titles and links
        """
        try:
            return self._search_papers(user_query)
        except Exception as e:
            return f"Error processing query: {str(e)}"


# Example usage
if __name__ == "__main__":
    from langchain_ollama import OllamaLLM
    
    # Initialize LLM for testing
    llm = OllamaLLM(model="gemma3:12b", temperature=0)
    
    # Create WebSearcher instance with the LLM
    searcher = WebSearcher(llm)
    
    print("=" * 60)
    print("Academic Paper Search Assistant")
    print("Powered by WebSearcher Class with LLM Organization")
    print("=" * 60)
    print("\nSearches for research papers and organizes them using LLM")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            user_query = input("\n🔍 Enter your research query: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Happy researching! 📚")
                break
            
            if not user_query:
                print("Please enter a valid query.")
                continue
            
            print("\n🤖 Searching and organizing papers...\n")
            result = searcher.search(user_query)
            print(f"\n📄 Results:\n{result}")
            print("\n" + "-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Happy researching! 📚")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different query.")