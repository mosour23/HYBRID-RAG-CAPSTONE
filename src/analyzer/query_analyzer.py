import spacy

class QueryAnalyzer:
    def __init__(self):
        # Load a lightweight spaCy model for fast English language processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Error: Please download the model using: python -m spacy download en_core_web_sm")
            self.nlp = None

    def calculate_semantic_density(self, query: str) -> float:
        """
        Compute semantic density as a percentage (number of entities / number of valid words)
        """
        if not self.nlp or not query.strip():
            return 0.0

        doc = self.nlp(query)
        entities = doc.ents
        num_entities = len(entities)

        valid_words = [token for token in doc if not token.is_punct and not token.is_space]
        num_words = len(valid_words)

        if num_words == 0:
            return 0.0

        semantic_density = num_entities / num_words
        return round(semantic_density, 3)

    def calculate_multihop_requirement(self, query: str) -> float:
        """
        Compute multi-hop reasoning requirements.
        """
        if not self.nlp or not query.strip():
            return 0.0

        doc = self.nlp(query.lower())
        
        multihop_indicators = {
            "and", "or", "but", "if", "because", "than", 
            "after", "before", "compare", "difference", "between"
        }
        
        indicator_count = 0
        
        for token in doc:
            if token.text in multihop_indicators or token.dep_ == "cc":
                indicator_count += 1

        k = 2.0
        score = min(indicator_count / k, 1.0)
        return round(score, 3)