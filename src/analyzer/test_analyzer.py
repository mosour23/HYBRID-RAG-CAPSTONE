from query_analyzer import QueryAnalyzer

def run_tests():
    print("Initializing Query Analyzer... Loading NLP Model...")
    analyzer = QueryAnalyzer()
    
    # A graded set of questions for system testing
    test_queries = [
        # Query 1: very simple (general conversation)
        "Hello, how are you doing today?",
        
        # Query 2: medium complexity (direct but contains technical terms)
        "What is the function of the OP-RAG module in the new architecture?",
        
        # Query 3: high complexity (multi-step and detail-heavy)
        "Compare the inference latency of the Long-Context model with the RAG pipeline when processing adversarial noise in Q3 of 2024."
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*40}")
        print(f"Test Case {i}:")
        print(f"Query: {query}")
        print(f"{'-'*40}")
        
        # Compute the variables
        sd_score = analyzer.calculate_semantic_density(query)
        mh_score = analyzer.calculate_multihop_requirement(query)
        
        print(f"\nFinal Outputs for Router:")
        print(f"-> Semantic Density (SD): {sd_score}")
        print(f"-> Multi-hop Requirement (MHR): {mh_score}")

if __name__ == "__main__":
    run_tests()