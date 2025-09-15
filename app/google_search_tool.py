from typing import List, Dict
from googlesearch import search  # pip install googlesearch-python

def google_search(query: str, num_results: int = 5, lang: str = "en") -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for r in search(query, num_results=num_results, lang=lang, advanced=True):
        items.append({
            "title": getattr(r, "title", "") or r.url,
            "url": r.url,  # IMPORTANT: key is 'url'
            "snippet": getattr(r, "description", "")
        })
    return items
