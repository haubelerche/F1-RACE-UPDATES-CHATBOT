# app/main.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Type, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console

from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.tools.base import ToolException

# Local imports (package-relative)
from .google_search_tool import google_search
from .data_io import load_race_history



# ---------- Setup ----------
load_dotenv()
console = Console()

SYSTEM_RULES = (
    "You are an intelligent F1 chatbot. Answer only Formula 1 questions.\n"
    "If a question is not about F1, reply exactly:\n"
    "\"I'm an F1 chatbot and can only answer Formula 1-related questions. Please ask me about F1.\""
)

# ---------- Helpers ----------
def _filter_race_history(query: str, race_history: List[Dict]) -> str:
    """
    Lọc nhanh race_history theo năm xuất hiện trong query (YYYY) để thêm ngữ cảnh.
    """
    years = set(re.findall(r"\b(19|20)\d{2}\b", query))
    if not years:
        # fallback: thử tìm các số 4 chữ số
        years = set(re.findall(r"\b\d{4}\b", query))
    if not years:
        return ""

    relevant_entries = []
    for e in race_history:
        y = str(e.get("year", ""))
        if y and any(y.endswith(yr) for yr in years):
            relevant_entries.append(e)

    if not relevant_entries:
        return ""

    # Cắt ngắn để tránh prompt quá dài
    lines = []
    for e in relevant_entries[:50]:
        lines.append(str(e))
    context = "\n".join(lines)
    return context[:2000]  # giới hạn an toàn

# ---- Google Search Tool ----
class GoogleSearchInput(BaseModel):
    query: str

class GoogleSearchTool(BaseTool):
    name: str = "google_search"
    description: str = "Search Google for real-time Formula 1 information."
    args_schema: Type[BaseModel] = GoogleSearchInput

    def _run(self, query: str) -> str:
        try:
            results = google_search(query)
            if not results:
                return "No results found."
            return "\n\n".join(
                f"{item.get('title','(no title)')}: {item.get('snippet','')}"
                f" (URL: {item.get('url','')})"
                for item in results
            )
        except Exception as e:
            # Trả về ToolException để agent nắm bắt
            raise ToolException(f"Google Search failed: {e}")

    async def _arun(self, query: str) -> str:
        # Optional async path if your tool supports it
        return self._run(query)

# ---------- Chatbot wrapper ----------
class F1Chatbot:
    def __init__(
        self,
        llm: ChatOpenAI,
        tools: List[BaseTool],
        race_history: List[Dict],
    ) -> None:
        self._llm = llm
        self._race_history = race_history
        self._memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            return_messages=True,
        )

        self._agent = initialize_agent(
            tools=tools,
            llm=self._llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=False,
            memory=self._memory,
            agent_kwargs={
                "system_message": SYSTEM_RULES,
                "extra_prompt_messages": [
                    MessagesPlaceholder(variable_name="chat_history")
                ],
            },
        )

        self._initialized = True

    # ---- Expected by Streamlit UI ----
    def chat(self, prompt: str) -> str:
        """
        Thực thi 1 lượt chat. Nếu tìm được context từ race_history, ghép vào prompt.
        """
        try:
            context = _filter_race_history(prompt, self._race_history)
            augmented_prompt = (
                f"{prompt}\n\n"
                f"[Context: Relevant race history snippets]\n{context}"
                if context else prompt
            )

            result = self._agent.invoke({"input": augmented_prompt})
            return (result.get("output") or "").strip()
        except Exception as e:
            raise RuntimeError(f"Agent error: {e}") from e

    def get_system_info(self) -> Dict[str, Union[str, int, bool]]:
        return {
            "model_device": "OpenAI API (remote)",
            "documents_loaded": len(self._race_history),
            "initialized": self._initialized,
        }

# ---------- Factory (used by Streamlit) ----------
# in app/main.py

def create_chatbot(
    data_path: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    model: str = None,               # accept None and resolve below
    temperature: float = 0.7,
) -> F1Chatbot:
    """
    Factory for the Streamlit UI. Returns an instance with .chat() and .get_system_info().
    """
    # Resolve API key
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Please set it in your environment or pass api_key=..."
        )

    # ---- Resolve model with env override + sensible default
    # You can set CHAT_MODEL=gpt-4o or any other in your env to override.
    model = model or os.getenv("CHAT_MODEL") or "gpt-4o-mini"
    fallback_models = [m for m in [
        os.getenv("CHAT_MODEL_FALLBACK"),
        "gpt-4.1-mini",
        "gpt-3.5-turbo",   # last resort if your account still has it
    ] if m]

    # ---- Load race history (see section B for the path fix)
    base = Path(data_path) if data_path else Path(__file__).resolve().parents[1]
    if (base / "race_history.jsonl").exists():
        race_history_file = base / "race_history.jsonl"
    else:
        race_history_file = base / "data" / "race_history.jsonl"
    race_history = load_race_history(str(race_history_file))

    # ---- Build LLM with graceful fallback on 403/model_not_found
    def _build_llm(model_name: str) -> ChatOpenAI:
        return ChatOpenAI(api_key=resolved_key, temperature=temperature, model=model_name)

    try:
        llm = _build_llm(model)
    except Exception as e:
        # If it's a model access error, try fallbacks
        msg = str(e)
        if "model_not_found" in msg or "does not have access to model" in msg or "404" in msg or "403" in msg:
            for fm in fallback_models:
                try:
                    llm = _build_llm(fm)
                    console.print(f"[bold yellow]Model '{model}' unavailable. Falling back to '{fm}'.[/bold yellow]")
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(f"Unable to initialize any model. Tried: {model}, {fallback_models}. Error: {e}") from e
        else:
            raise

    # Tools
    tools = [GoogleSearchTool()]

    return F1Chatbot(llm=llm, tools=tools, race_history=race_history)


# ---------- Optional CLI runner ----------
def main() -> None:
    """
    Simple CLI loop for local testing:
    python -m app.main
    """
    console.print("[bold green]Welcome to the F1 Chatbot![/bold green]")
    console.print("Type [bold yellow]'exit'[/bold yellow] to end.\n")

    try:
        bot = create_chatbot()
    except Exception as e:
        console.print(f"[bold red]Initialization error:[/bold red] {e}")
        return

    while True:
        user_input = console.input("[bold yellow]You:[/bold yellow] ")
        if user_input.strip().lower() == "exit":
            console.print("[bold green]Goodbye![/bold green]")
            break
        try:
            reply = bot.chat(user_input)
            console.print(f"[bold green]Bot:[/bold green] {reply}")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    main()
