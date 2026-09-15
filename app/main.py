"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.db.database import check_database

app = FastAPI(title="Enterprise AI Lab", version="0.1.0")


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Return the API liveness status."""
    database_status = "ok" if await check_database() else "error"

    return {
        "api": "ok",
        "database": database_status,
    }


app.include_router(documents_router)
app.include_router(search_router)
app.include_router(chat_router)
