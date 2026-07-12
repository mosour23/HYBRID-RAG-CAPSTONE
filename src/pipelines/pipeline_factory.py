from src.pipelines.base_strategy import RetrievalStrategy
from src.pipelines.rag_pipeline import RAGPipeline
from src.pipelines.lc_pipeline import LongContextPipeline

class PipelineFactory:
    _rag_instance = None
    _lc_instance = None

    @classmethod
    def get_pipeline(cls, decision: str) -> RetrievalStrategy:
        """
        Factory method to return the correct pipeline based on the router's decision.
        Implements Singleton-like behavior to avoid re-initializing models.
        """
        if decision == "OP-RAG":
            if cls._rag_instance is None:
                cls._rag_instance = RAGPipeline()
            return cls._rag_instance
            
        elif decision == "Long-Context":
            if cls._lc_instance is None:
                cls._lc_instance = LongContextPipeline()
            return cls._lc_instance
            
        raise ValueError(f"Unknown routing decision: {decision}")