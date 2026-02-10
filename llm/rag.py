import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class RAGAssistant:
    def __init__(self, metadata_path: str):
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def build_context(self) -> str:
        context = f"""
Model Type: {self.metadata['model_type']}
Test Size: {self.metadata['test_size']}
Metrics:
"""
        for k, v in self.metadata["metrics"].items():
            context += f"\n{k}: {v}"
        return context

    def ask(self, question: str) -> str:
        try:
            context = self.build_context()

            prompt = f"""
You are an expert compensation analyst and data scientist explaining salary prediction results.
The model predicts the average annual salary based on job market attributes.

Domain Context:
- Skills: Python, R, Spark, AWS, Excel.
- Attributes: Job title (simplified), Seniority, Company Rating, Sector, Industry.
- Model Info: {context}

Goal: Provide insights into how features like "Python usage" or "Seniority" might influence salary predictions based on the model's metadata and typical job market trends.

Question:
{question}

Answer clearly, using the context provided. If the question is about specific skill impacts, explain them in the context of the model's reported metrics.
"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"RAG Error: {e}")
            raise e
