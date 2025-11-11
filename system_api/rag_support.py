
from sentence_transformers import SentenceTransformer

import faiss
from pathlib import Path
import PyPDF2
import requests
import re
class RAGChatbot:

    def __init__(self, data_dir="data", ollama_model="gemma3:12b"):
        """Initialize RAG Chatbot with Ollama"""
        self.data_dir = Path(data_dir)
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        self.documents = []
        self.embeddings = None
        self.index = None
        
        self._check_ollama()
        
    def _check_ollama(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.post(
                "http://localhost:11434/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if not any(self.ollama_model in name for name in model_names):
                    print(f"Warning: Model {self.ollama_model} not found.")
                    print(f"Available models: {model_names}")
                    print(f"Run: ollama pull {self.ollama_model}")
            else:
                print("Warning: Could not connect to Ollama")
        except requests.exceptions.RequestException:
            print("Error: Ollama is not running. Start it with: ollama serve")
            
    def load_pdfs(self):
        """Load and process all PDF files from data directory"""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
            print(f"Created {self.data_dir} directory. Please add PDF files there.")
            return
        
        pdf_files = list(self.data_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {self.data_dir}")
            return
        
        print(f"Loading {len(pdf_files)} PDF files...")
        for pdf_file in pdf_files:
            try:
                text = self._extract_text_from_pdf(pdf_file)
                chunks = self._split_into_chunks(text, chunk_size=500)
                for chunk in chunks:
                    if chunk.strip():
                        self.documents.append({
                            'text': chunk,
                            'source': pdf_file.name
                        })
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
        
        print(f"Loaded {len(self.documents)} document chunks")
        
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def _split_into_chunks(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def create_index(self):
        """Create FAISS index from documents"""
        if not self.documents:
            print("No documents loaded. Run load_pdfs() first.")
            return
        
        print("Creating embeddings...")
        texts = [doc['text'] for doc in self.documents]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Index created with {self.index.ntotal} vectors")
    
    def retrieve_relevant_docs(self, query, top_k=3):
        """Retrieve most relevant documents for a query"""
        if self.index is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        relevant_docs = []
        for idx in indices[0]:
            if idx < len(self.documents):
                relevant_docs.append(self.documents[idx])
        
        return relevant_docs
    
    def generate_response(self, query, context_docs):
        """Generate response using Ollama with retrieved context"""
        context = "\n\n".join([
            f"[Source: {doc['source']}]\n{doc['text']}" 
            for doc in context_docs
        ])
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the user's question. If the answer cannot be found in the context, say so.
        Context:
        {context}

        Question: {query}

        Answer:"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                raw = response.json()['response']

                # Detect and strip <think> blocks
                think_pattern = re.compile(r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL)
                has_think = bool(think_pattern.search(raw))
                clean = think_pattern.sub("", raw).strip()

                # Extra hardening: collapse >2 blank lines
                clean = re.sub(r"\n{3,}", "\n\n", clean)

                return {"text": clean, "reasoning_hidden": has_think}
            else:
                return {"text": f"Error: {response.status_code} - {response.text}", "reasoning_hidden": False}
                    
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {e}"
    
    def chat(self, query):
        """Main chat function - retrieve and generate"""
        if self.index is None:
            return None, "Please load documents and create index first."
        
        relevant_docs = self.retrieve_relevant_docs(query, top_k=3)
        
        if not relevant_docs:
            return [], "No relevant documents found."
        
        result = self.generate_response(query, relevant_docs)
        sources = list(set([doc['source'] for doc in relevant_docs]))
        return sources, result  # result is a dict {text, reasoning_hidden}

