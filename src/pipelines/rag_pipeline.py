import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

class RAGPipeline:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', llm_model: str = "llama-3.1-8b-instant"):
        print(f"Loading Embedding Model ({embedding_model})...")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Work around the legacy warning (FutureWarning)
        try:
            self.dimension = self.embedder.get_embedding_dimension()
        except AttributeError:
            self.dimension = self.embedder.get_sentence_embedding_dimension()
            
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        
        # Initialize the Groq model connection
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Error: GROQ_API_KEY not found in .env file.")
        self.client = Groq(api_key=api_key)
        self.llm_model = llm_model

    def ingest_documents(self, docs: list[str]):
        if not docs: return
        print(f"Ingesting {len(docs)} documents into FAISS index...")
        embeddings = self.embedder.encode(docs)
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
        
        # The list automatically tracks the original chronological order based on the index
        self.documents.extend(docs)
        print(f"Total documents in index: {self.index.ntotal}")

    def retrieve_and_generate(self, query: str, top_k: int = 2) -> str:
        if self.index.ntotal == 0:
            return "Error: The database is empty."

        # 1. Fast retrieval from FAISS
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        # 2. Extract valid indices
        valid_indices = [idx for idx in indices[0] if idx != -1 and idx < len(self.documents)]
        
        # 3. OP-RAG CORE LOGIC: Sort indices to preserve the original chronological order
        sorted_indices = sorted(valid_indices)

        # 4. Fetch the documents in their original order
        retrieved_docs = []
        for idx in sorted_indices:
            retrieved_docs.append(self.documents[idx])
                
        # 5. Build the ordered context
        context = "\n".join([f"- {doc}" for doc in retrieved_docs])
        
        # 6. Fast, direct generation via Llama-3
        print("⚡ Generating fast factual response via Llama-3 (OP-RAG enabled)...")
        prompt = f"Use ONLY the following context to answer the query briefly.\n\nContext:\n{context}\n\nQuery: {query}\nAnswer:"
        
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a fast, factual assistant. Answer strictly based on the provided context in 1 or 2 clear sentences."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content