#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
