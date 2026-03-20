import logging

from .base import Tool

logger = logging.getLogger(__name__)

MAX_CONTENT_LENGTH = 4000


class WebFetchTool(Tool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch a URL and extract its main text content."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
            },
            "required": ["url"],
        }

    async def execute(self, **kwargs) -> str:
        url = kwargs.get("url", "")
        if not url:
            return "Error: No URL provided."

        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=15.0,
                headers={"User-Agent": "Mozilla/5.0 (compatible; InferenceAPI/1.0)"},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove non-content elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            # Collapse multiple blank lines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)

            if len(text) > MAX_CONTENT_LENGTH:
                text = text[:MAX_CONTENT_LENGTH] + "\n...[truncated]"

            return text if text else "No readable content found at this URL."

        except Exception as e:
            logger.error(f"Web fetch failed for {url}: {e}")
            return f"Fetch error: {str(e)}"
