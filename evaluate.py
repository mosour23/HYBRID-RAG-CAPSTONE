import json
from src.pipelines.pipeline_factory import PipelineFactory

def calculate_kpr(generated_answer: str, key_points: list[str], poison_terms: list[str] = None) -> float:
    """
    مقياس KPR المطور: يحسب نسبة تذكر المعلومات الصحيحة + نسبة رفض المعلومات المسممة
    """
    if not key_points: 
        return 0.0
        
    ans = generated_answer.lower()
    
    # 1. Compute the correct-point recall rate
    recall = sum(1 for p in key_points if p.lower() in ans) / len(key_points)
    
    # 2. Compute the noise/toxin rejection rate
    if poison_terms:
        leaked = sum(1 for p in poison_terms if p.lower() in ans)
        rejection = 1 - (leaked / len(poison_terms))
        return round(((recall + rejection) / 2) * 100, 2)
        
    return round(recall * 100, 2)

def load_dataset(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def run_real_data_evaluation():
    print("="*60)
    print("📊 Starting AUTOMATED System Evaluation (Real Dataset)")
    print("="*60)

    # Use the Factory Pattern to instantiate the pipelines
    rag_pipeline = PipelineFactory.get_pipeline("OP-RAG")
    lc_pipeline = PipelineFactory.get_pipeline("Long-Context")

    dataset = load_dataset('data/test_dataset.json')
    corpus = dataset['corpus']
    test_cases = dataset['test_cases']

    rag_pipeline.ingest_documents(corpus)
    full_lc_context = "\n".join(corpus)

    results = {"RAG": {"latency": [], "kpr": []}, "LC": {"latency": [], "kpr": []}}

    print(f"\n🚀 Running {len(test_cases)} test cases...\n")
    
    for i, test in enumerate(test_cases, 1):
        query = test["query"]
        ground_truth = test["ground_truth"]
        poison_terms = test.get("poison_terms", None)
        
        print(f"--- Test Case {i}: {query} ---")
        
        # --- RAG pipeline test ---
        rag_ans, rag_ttft = rag_pipeline.generate(query)
        rag_kpr = calculate_kpr(rag_ans, ground_truth, poison_terms)
        
        results["RAG"]["latency"].append(rag_ttft)
        results["RAG"]["kpr"].append(rag_kpr)
        print(f"⚡ RAG -> TTFT: {rag_ttft:.3f}s | KPR: {rag_kpr}%")

        # --- LC pipeline test ---
        lc_ans, lc_ttft = lc_pipeline.generate(query, context=full_lc_context)
        lc_kpr = calculate_kpr(lc_ans, ground_truth, poison_terms)
        
        results["LC"]["latency"].append(lc_ttft)
        results["LC"]["kpr"].append(lc_kpr)
        print(f"🧠 LC  -> TTFT: {lc_ttft:.3f}s | KPR: {lc_kpr}%\n")

    avg_rag_lat = sum(results["RAG"]["latency"]) / len(test_cases) if test_cases else 0
    avg_rag_kpr = sum(results["RAG"]["kpr"]) / len(test_cases) if test_cases else 0
    
    avg_lc_lat = sum(results["LC"]["latency"]) / len(test_cases) if test_cases else 0
    avg_lc_kpr = sum(results["LC"]["kpr"]) / len(test_cases) if test_cases else 0

    print("="*60)
    print("🏆 FINAL EVALUATION REPORT (AVERAGES)")
    print("="*60)
    print(f"RAG Pipeline -> Avg TTFT: {avg_rag_lat:.3f}s | Avg KPR: {avg_rag_kpr:.1f}%")
    print(f"LC Pipeline  -> Avg TTFT: {avg_lc_lat:.3f}s | Avg KPR: {avg_lc_kpr:.1f}%")
    
    if avg_rag_lat > 0:
        print(f"\n💡 Conclusion: RAG is {avg_lc_lat / avg_rag_lat:.1f}x faster on average.")
    print("="*60)

if __name__ == "__main__":
    run_real_data_evaluation()