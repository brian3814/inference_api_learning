import base64
import logging
from io import BytesIO

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(url: str) -> Image.Image:
    """Load an image from a base64 data URL or HTTP URL."""
    if url.startswith("data:"):
        _, encoded = url.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    else:
        response = httpx.get(url, timeout=15, follow_redirects=True)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
