import asyncio
import logging

from .base import Tool

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo and return top results with titles, snippets, and URLs."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> str:
        query = kwargs.get("query", "")
        if not query:
            return "Error: No search query provided."

        try:
            from duckduckgo_search import DDGS

            def _search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=5))

            results = await asyncio.to_thread(_search)

            if not results:
                return f"No results found for: {query}"

            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                body = r.get("body", "No snippet")
                href = r.get("href", "")
                formatted.append(f"{i}. {title}\n   {body}\n   URL: {href}")

            return "\n\n".join(formatted)

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Search error: {str(e)}"
