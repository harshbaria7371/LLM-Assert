import os
import google.generativeai as genai
from openai import OpenAI

def _resolve_gemini_model(requested: str | None):
    try:
        models = list[Any](genai.list_models())
    except Exception:
        return requested or "gemini-1.5-pro"
    
    def short_name(full_name: str) -> str:
        return full_name.split("/")[-1] if full_name else full_name
    
    # Only consider models that support generateContent.
    candidates = [
        m for m in models
        if "generateContent" in getattr(m, "supported_generation_methods", [])
    ]
    if not candidates:
        return requested or "gemini-1.5-pro"
    
    # If a requested name is provided, try to match it.
    if requested:
        for m in candidates:
            full = getattr(m, "name", "")
            if requested == full or requested == short_name(full):
                return short_name(full) or requested
            
    first = candidates[0]
    return short_name(getattr(first, "name", "")) or requested or "gemini-1.5-pro"

class LLMClient:
    def __init__(self, model: str | None = None):
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        target_model = model or os.getenv("TARGET_MODEL", "gemini-2.5-flash")

        self.provider = provider
        self.model = target_model

        if self.provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set but LLM_PROVIDER is Gemini")
            genai.configure(api_key=api_key)

            resolved_model = _resolve_gemini_model(self.model)
            if resolved_model != self.model:
                print(
                    f"[LLMClient] Gemini Model '{self.model}' not found. ",
                    f"using '{resolved_model}' instead. ",
                    flush=True
                )
            self.model = resolved_model

            self.client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Invalid LLM Provider: {self.provider}")
        
    def ask(self, prompt: str, temperature=0.7):
        try:
            if self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=30,
                )
                return resp.choices[0].message.content
            elif self.provider == "gemini":
                resp = self.client.generate_content(
                    prompt,
                    generation_config={"temperature": float(temperature)},
                )
                return resp.text
        except Exception as e:
            raise