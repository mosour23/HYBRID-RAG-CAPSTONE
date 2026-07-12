import time
import json
from src.pipelines.rag_pipeline import RAGPipeline
from src.pipelines.lc_pipeline import LongContextPipeline

def calculate_kpr(generated_answer: str, key_points: list[str], poison_terms: list[str] = None) -> float:
    """
    مقياس KPR المطور: يحسب نسبة تذكر المعلومات الصحيحة + نسبة رفض المعلومات المسممة
    """
    if not key_points: 
        return 0.0
        
    ans = generated_answer.lower()
    
    # 1. حساب نسبة النقاط الصحيحة (Recall)
    recall = sum(1 for p in key_points if p.lower() in ans) / len(key_points)
    
    # 2. حساب نسبة رفض الضوضاء/السموم (Rejection)
    if poison_terms:
        leaked = sum(1 for p in poison_terms if p.lower() in ans)
        rejection = 1 - (leaked / len(poison_terms))
        # المتوسط بين الاسترجاع الصحيح ورفض السموم
        return round(((recall + rejection) / 2) * 100, 2)
        
    return round(recall * 100, 2)

def load_dataset(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def run_real_data_evaluation():
    print("="*60)
    print("📊 Starting AUTOMATED System Evaluation (Real Dataset)")
    print("="*60)

    # 1. تهيئة النظام
    rag_pipeline = RAGPipeline()
    lc_pipeline = LongContextPipeline()

    # 2. تحميل البيانات الحقيقية من مجلد data
    dataset = load_dataset('data/test_dataset.json')
    corpus = dataset['corpus']
    test_cases = dataset['test_cases']

    # 3. إدخال كل النصوص دفعة واحدة (Ingestion)
    rag_pipeline.ingest_documents(corpus)
    full_lc_context = "\n".join(corpus)

    # متغيرات لتخزين النتائج النهائية لحساب المتوسطات
    results = {"RAG": {"latency": [], "kpr": []}, "LC": {"latency": [], "kpr": []}}

    # 4. حلقة اختبار أوتوماتيكية تمر على كل الأسئلة
    print(f"\n🚀 Running {len(test_cases)} test cases...\n")
    
    for i, test in enumerate(test_cases, 1):
        query = test["query"]
        ground_truth = test["ground_truth"]
        
        print(f"--- Test Case {i}: {query} ---")
        
        # --- اختبار مسار RAG ---
        start_time = time.time()
        rag_ans = rag_pipeline.retrieve_and_generate(query)
        rag_lat = time.time() - start_time
        rag_kpr = calculate_kpr(rag_ans, ground_truth)
        
        results["RAG"]["latency"].append(rag_lat)
        results["RAG"]["kpr"].append(rag_kpr)
        print(f"⚡ RAG -> Latency: {rag_lat:.3f}s | KPR: {rag_kpr}%")

        # --- اختبار مسار LC ---
        start_time = time.time()
        lc_ans = lc_pipeline.process_full_context(query, full_lc_context)
        lc_lat = time.time() - start_time
        lc_kpr = calculate_kpr(lc_ans, ground_truth)
        
        results["LC"]["latency"].append(lc_lat)
        results["LC"]["kpr"].append(lc_kpr)
        print(f"🧠 LC  -> Latency: {lc_lat:.3f}s | KPR: {lc_kpr}%\n")

    # 5. التقرير النهائي (Averages)
    avg_rag_lat = sum(results["RAG"]["latency"]) / len(test_cases)
    avg_rag_kpr = sum(results["RAG"]["kpr"]) / len(test_cases)
    
    avg_lc_lat = sum(results["LC"]["latency"]) / len(test_cases)
    avg_lc_kpr = sum(results["LC"]["kpr"]) / len(test_cases)

    print("="*60)
    print("🏆 FINAL EVALUATION REPORT (AVERAGES)")
    print("="*60)
    print(f"RAG Pipeline -> Avg Latency: {avg_rag_lat:.3f}s | Avg KPR: {avg_rag_kpr:.1f}%")
    print(f"LC Pipeline  -> Avg Latency: {avg_lc_lat:.3f}s | Avg KPR: {avg_lc_kpr:.1f}%")
    print(f"\n💡 Conclusion: RAG is {avg_lc_lat / avg_rag_lat:.1f}x faster on average.")
    print("="*60)

if __name__ == "__main__":
    run_real_data_evaluation()