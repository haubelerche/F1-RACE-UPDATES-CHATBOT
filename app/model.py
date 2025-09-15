from typing import Optional
import logging
import os

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

logger = logging.getLogger(__name__)


class F1ChatbotModel:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self._openai_client = None

    def load_model(self):
        """Initialize the OpenAI client using API key from environment."""
        if self._openai_client is not None:
            return
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Please install 'openai'.")
        # Map legacy var name if present
        if "OPENAI_API_KEY" not in os.environ and os.getenv("OPEN_AI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.environ["OPEN_AI_API_KEY"]
        self._openai_client = OpenAI()
        logger.info(f"OpenAI client initialized for model: {self.model_name}")

    def generate_response(self, context: str, question: str, max_length: int = 512) -> str:
        if self._openai_client is None:
            self.load_model()
        try:
            system_prompt = (
                "You are an F1 expert assistant. Be precise and concise. "
                "Use the provided context to answer questions about Formula 1."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ]
            completion = self._openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=max_length,
            )
            text = completion.choices[0].message.content or ""
            return text.strip()
        except Exception as e:
            logger.error(f"Error generating response via OpenAI: {e}")
            return "I'm sorry, I encountered an error while processing your question."