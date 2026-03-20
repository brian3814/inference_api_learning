from .base import tool_registry
from .web_search import WebSearchTool
from .web_fetch import WebFetchTool

tool_registry.register(WebSearchTool())
tool_registry.register(WebFetchTool())

__all__ = ["tool_registry"]
