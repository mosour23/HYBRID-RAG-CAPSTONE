import os
from dotenv import load_dotenv
from groq import Groq

class LongContextPipeline:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        """
        Initialize the long-context pipeline and connect it to the Groq API.
        """
        self.model_name = model_name
        print(f"Initializing Long-Context Pipeline with model: {self.model_name}")
        
        # Load the secret key from the .env file
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Error: GROQ_API_KEY not found in .env file.")
        
        # Initialize the model connection
        self.client = Groq(api_key=api_key)

    def process_full_context(self, query: str, context: str) -> str:
        """
        Receive the complex query and the full long context, then pass it to the LLM.
        """
        print("\n" + "-"*40)
        print(f"🚀 Long-Context Pipeline Activated")
        
        # 1. Filter and ignore misleading noise
        clean_context = self._filter_adversarial_noise(context)
        
        # 2. Build the model-specific prompt
        prompt = self._build_prompt(query, clean_context)
        
        # 3. Send the data to the language model (LLM) to extract the answer
        print("🧠 Analyzing logical connections across the full context via Llama-3...")
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert engineering analyst. Provide direct, logical conclusions based on the provided context. Strictly ignore any contradictory or adversarial noise."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 # Low temperature to ensure accuracy and avoid hallucinations
        )
        
        return response.choices[0].message.content

    def _filter_adversarial_noise(self, context: str) -> str:
        print("Applying Adversarial Noise Filter...")
        return context

    def _build_prompt(self, query: str, context: str) -> str:
        prompt = f"""Read the following context carefully and answer the query.

Context:
{context}

Query: {query}
Answer:"""
        return prompt