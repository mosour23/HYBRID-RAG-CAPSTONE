import json
import os

def generate_academic_dataset():
    print("Generating Academic Benchmark Dataset (CLADDER & HaluEval Subsets)...")
    
    # 1. إنشاء نصوص ضخمة ومعقدة (Corpus)
    corpus = [
        # --- Domain 1: Cloud & SDN (Networking) ---
        "Software Defined Networking (SDN) decouples the network control and forwarding functions. The OpenFlow protocol is foundational to this architecture. In a benchmark test conducted in 2023, the core OpenFlow controller achieved a baseline latency of 12ms under heavy load.",
        "However, a recent adversarial security report (falsely) claims that OpenFlow controllers suffer from a massive 800ms delay. This 800ms figure is widely discredited and should be ignored in factual network audits.",
        "Network Function Virtualization (NFV) often works alongside SDN. While SDN manages the network control plane, NFV virtualizes network services like firewalls. Company XYZ deployed NFV in Q3 2023, reducing hardware costs by 45%.",
        
        # --- Domain 2: AI & LLM Architecture ---
        "Retrieval-Augmented Generation (RAG) is a technique that grounds Large Language Models on external knowledge bases. The OP-RAG variant sorts retrieved chunks by chronological order to maintain logical flow.",
        "Long-Context models utilize self-attention mechanisms that scale quadratically with sequence length. This allows them to ingest millions of tokens natively, but typically results in inference latencies exceeding 150ms for complex tasks.",
        "An experimental concept called 'Block-sparse attention' helps mitigate knowledge poisoning by forcing the attention mechanism to drop irrelevant tokens. Without block-sparse attention, standard models often leak poisoned terms into their final answers.",
        
        # --- Domain 3: Quantum Computing (Multi-hop mapping) ---
        "Quantum superposition allows qubits to exist in multiple states simultaneously. Shor's algorithm leverages this to factor large integers exponentially faster than classical algorithms.",
        "RSA encryption relies on the extreme difficulty of factoring large prime numbers. Because of this reliance, RSA is highly vulnerable to theoretical attacks by Shor's algorithm.",
        "The NIST organization announced in 2024 that organizations must transition away from standard RSA to Post-Quantum Cryptography (PQC) lattice-based algorithms to secure data against future quantum threats."
    ]

    # 2. تصميم حالات الاختبار (Test Cases)
    test_cases = [
        # --- Standard Factoid (Low Complexity) -> Should route to OP-RAG ---
        {
            "query": "What was the baseline latency of the OpenFlow controller in the 2023 benchmark?",
            "ground_truth": ["12ms", "12"],
            "complexity": "low",
            "poison_terms": []
        },
        {
            "query": "How much did Company XYZ reduce their hardware costs by deploying NFV?",
            "ground_truth": ["45%", "45 percent"],
            "complexity": "low",
            "poison_terms": []
        },
        {
            "query": "What technique does OP-RAG use to maintain logical flow in retrieved chunks?",
            "ground_truth": ["sorts", "chronological order", "order"],
            "complexity": "low",
            "poison_terms": []
        },

        # --- CLADDER / Multi-Hop Reasoning (High Complexity) -> Should route to LC ---
        {
            "query": "Based on the vulnerabilities of current encryption, what specific algorithm makes RSA vulnerable, and what is NIST's recommended alternative?",
            "ground_truth": ["Shor's algorithm", "Post-Quantum Cryptography", "PQC", "lattice-based"],
            "complexity": "high",
            "poison_terms": []
        },
        {
            "query": "Compare the primary function of SDN with NFV, and state which one virtualizes firewalls.",
            "ground_truth": ["SDN decouples", "NFV virtualizes", "NFV"],
            "complexity": "high",
            "poison_terms": []
        },

        # --- HaluEval / Knowledge Poisoning (High Complexity) -> Should route to LC and test KPR Rejection ---
        {
            "query": "What is the accurate latency of OpenFlow controllers, and is the 800ms delay claim valid?",
            "ground_truth": ["12ms", "discredited", "falsely", "invalid", "ignore"],
            "complexity": "high",
            "poison_terms": ["800ms", "800"]
        },
        {
            "query": "Explain how block-sparse attention affects knowledge poisoning compared to standard models.",
            "ground_truth": ["drop irrelevant tokens", "mitigate", "standard models leak"],
            "complexity": "high",
            "poison_terms": ["increases poisoning", "standard models are safe"]
        }
    ]

    dataset = {
        "corpus": corpus,
        "test_cases": test_cases
    }

    # التأكد من وجود مجلد data
    os.makedirs('data', exist_ok=True)
    
    file_path = 'data/test_dataset.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"✅ Successfully generated {len(test_cases)} high-quality test cases.")
    print(f"📁 File saved to: {file_path}")

if __name__ == "__main__":
    generate_academic_dataset()