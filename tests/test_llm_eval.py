import os
import pytest
import json
from utils.llm_client import LLMClient
from utils.evaluators import Evalutor

with open("data/test_cases.json") as f:
    test_data = json.load(f)

USE_EVALUTOR = os.getenv("USE_EVALUTOR", "true").lower() == "true"

@pytest.mark.parametrize("case", test_data)
def test_llm_performance(case):
    client = LLMClient()
    evaluator = Evalutor()

    print("-"*20)
    print(f"Prompt: {case['prompt']}", flush=True)

    response = client.ask(case["prompt"])

    print(f"Response: {response}", flush=True)
    print("-"*20)

    # Verification Logic
    if USE_EVALUTOR:
        if case["evaluation_type"] == "exact_match":
            is_correct = evaluator.evalute_logic(response, case["expected_answer"])
            assert is_correct, f"Logic Fail! Expected {case['expected_answer']} in response: {response}"

        elif case["evaluation_type"] == "llm_judge":
            result = evaluator.evaluate_with_judge(case["prompt"], response, case["criteria"])
            assert result["score"] >= 7, f"Judge failed the response! Reason: {result['reasoning']}"
    else:
        keywords = case.get("keywords", [])
        res_lower = (response or "").lower()
        missing = [ kw for kw in keywords if kw.lower() not in res_lower ]
        assert not missing, (
            f"Keyword check failed for case {case['id']}.",
            f"Missing keywords: {missing}. Response: {response}"
        )