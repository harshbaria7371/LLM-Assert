import re
import os
from utils.llm_client import LLMClient

class Evalutor:
    def __init__(self):
        judge_model = os.getenv("JUDGE_MODEL") or os.getenv("TARGET_MODEL")
        self.judge = LLMClient(model=judge_model)

    def evalute_logic(self, response: str, expected: str) -> bool:
        pattern = rf"\b{re.escape(expected)}\b"
        return bool(re.search(pattern, response))
    
    def evaluate_with_judge(self, prompt: str, ai_response: str, criteria: str) -> dict:
        
        judging_prompt = f"""
        You are an expert Quality Auditor. Evaluate the following AI response based on the criteria.

        Orignial Prompt: {prompt}
        AI Response: {ai_response}
        Evaluation Criteria: {criteria}

        Provide your response in this format:
        Score: [0-10]
        Reasoning: [Short explanation]
        """

        judgment = self.judge.ask(judging_prompt)

        # Simple parsing logic
        score_match = re.search(r"Score:\s*(\d+)", judgment)
        score = int(score_match.group(1)) if score_match else 0
        return {"score": score, "reasoning": judgment}
    
    