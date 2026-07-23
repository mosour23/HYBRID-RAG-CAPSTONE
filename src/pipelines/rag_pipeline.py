import os
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
try:
    from dotenv import load_dotenv
except Exception:
    # If python-dotenv is not installed, provide a no-op and warn.
    def load_dotenv():
        print("Warning: python-dotenv not installed; skipping .env load.")
try:
    from groq import Groq
except Exception:
    Groq = None
    print("Warning: groq package not installed; Groq client will be disabled.")
from src.pipelines.base_strategy import RetrievalStrategy

class RAGPipeline(RetrievalStrategy): 
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', llm_model: str = "llama-3.1-8b-instant"):
        print(f"Loading Embedding Model ({embedding_model})...")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Smart compatibility with older and newer versions of the library
        if hasattr(self.embedder, 'get_embedding_dimension'):
            self.dimension = self.embedder.get_embedding_dimension()
        else:
            self.dimension = self.embedder.get_sentence_embedding_dimension()
            
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or Groq is None:
            # Do not raise at import time; allow the app to start and provide
            # a helpful message when the pipeline is invoked.
            print("Warning: GROQ_API_KEY not found or groq unavailable; Groq client disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
        self.llm_model = llm_model

    def ingest_documents(self, docs: list[str]):
        if not docs: return
        print(f"Ingesting {len(docs)} documents into FAISS index...")
        embeddings = self.embedder.encode(docs)
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
        self.documents.extend(docs)
        print(f"Total documents in index: {self.index.ntotal}")

    def retrieve_and_generate(self, query: str, top_k: int = 2):
        if self.index.ntotal == 0:
            return "Error: The database is empty.", 0.0

        if self.client is None:
            return ("Error: Groq client not configured. Set GROQ_API_KEY in the environment and install python-dotenv (pip install python-dotenv).", 0.0)

        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        valid_indices = [idx for idx in indices[0] if idx != -1 and idx < len(self.documents)]
        sorted_indices = sorted(valid_indices)

        retrieved_docs = []
        for idx in sorted_indices:
            retrieved_docs.append(self.documents[idx])
                
        context = "\n".join([f"- {doc}" for doc in retrieved_docs])
        
        print("⚡ Generating fast factual response via Llama-3 (OP-RAG enabled)...")
        prompt = f"Use ONLY the following context to answer the query briefly.\n\nContext:\n{context}\n\nQuery: {query}\nAnswer:"
        
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a fast, factual assistant. Answer strictly based on the provided context in 1 or 2 clear sentences."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            stream=True 
        )
        
        ttft = None
        full_response = ""
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.time() - start_time 
                full_response += chunk.choices[0].delta.content
                
        return full_response, (ttft if ttft else 0.0)

    def generate(self, query: str, **kwargs) -> tuple[str, float]:
        return self.retrieve_and_generate(query)