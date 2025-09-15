# app/main.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Type, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console

from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.tools.base import ToolException

# Local imports
from .google_search_tool import google_search
from __future__ import annotations

from pathlib import Path

# ---------- Setup ----------
load_dotenv()
console = Console()

SYSTEM_RULES = (
    "You are an intelligent F1 chatbot. Answer only Formula 1 questions. "
    "If a question is not about F1, reply: "
    "\"I'm an F1 chatbot and can only answer Formula 1-related questions. Please ask me about F1.\""
)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def is_file(p: Path) -> bool:
    return p.is_file()

# ---------- Helpers ----------
def _filter_race_history(query: str, race_history: List[Dict]) -> str:
    relevant_entries = [e for e in race_history if str(e.get("year")) in query]
    if not relevant_entries:
        return ""
    return "\n".join(str(e) for e in relevant_entries)[:1000]  # keep context short


# ---- Google Search Tool
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
                f"{item['title']}: {item['snippet']} (URL: {item['url']})"
                for item in results
            )
        except Exception as e:
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
        # Add filtered race history as context when relevant
        context = _filter_race_history(prompt, self._race_history)
        inputs = {"input": prompt, "race_history": context}
        try:
            result = self._agent.invoke(inputs)
            # LangChain returns a dict with "output" for this agent type
            return result.get("output", "").strip()
        except Exception as e:
            raise RuntimeError(f"Agent error: {e}") from e

    def get_system_info(self) -> Dict[str, str | int | bool]:
        # Minimal info for your sidebar
        return {
            "model_device": "OpenAI API (remote)",
            "documents_loaded": len(self._race_history),
            "initialized": self._initialized,
        }


# ---------- Factory (used by Streamlit) ----------
def create_chatbot(
    data_path: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
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

    # Load race history
    base = Path(data_path) if data_path else Path(__file__).resolve().parents[1]
    race_history_file = base / "data" / "race_history.jsonl"
    race_history = load_race_history(str(race_history_file))

    # LLM
    llm = ChatOpenAI(
        api_key=resolved_key,
        temperature=temperature,
        model=model,
    )

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
