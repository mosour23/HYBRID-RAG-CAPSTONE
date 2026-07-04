from query_analyzer import QueryAnalyzer

def run_tests():
    print("Initializing Query Analyzer... Loading NLP Model...")
    analyzer = QueryAnalyzer()
    
    # قائمة بأسئلة متدرجة الصعوبة لاختبار النظام
    test_queries = [
        # استعلام 1: بسيط جداً (محادثة عامة)
        "Hello, how are you doing today?",
        
        # استعلام 2: متوسط التعقيد (مباشر لكن يحتوي مصطلحات)
        "What is the function of the OP-RAG module in the new architecture?",
        
        # استعلام 3: شديد التعقيد (متعدد الخطوات ومليء بالتفاصيل)
        "Compare the inference latency of the Long-Context model with the RAG pipeline when processing adversarial noise in Q3 of 2024."
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*40}")
        print(f"Test Case {i}:")
        print(f"Query: {query}")
        print(f"{'-'*40}")
        
        # حساب المتغيرات
        sd_score = analyzer.calculate_semantic_density(query)
        mh_score = analyzer.calculate_multihop_requirement(query)
        
        print(f"\nFinal Outputs for Router:")
        print(f"-> Semantic Density (SD): {sd_score}")
        print(f"-> Multi-hop Requirement (MHR): {mh_score}")

if __name__ == "__main__":
    run_tests()