"""
Opt-in API key auth.

Set CHATTERBOX_API_KEY in the environment to enable. When unset (default),
all endpoints remain open — preserves existing LAN-only behavior.

When enabled, every request must send `X-API-Key: <key>` or
`Authorization: Bearer <key>`. The following paths stay public regardless
of auth state so health checks and the static UI files keep working:

    /health, /, /stream-test, /studio, /favicon.ico

WebSockets are also checked (the key goes in the `api_key` query param or
the `X-API-Key` subprotocol header from the client).
"""
import hmac
import os
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse


PUBLIC_PATHS = {"/", "/health", "/stream-test", "/studio", "/favicon.ico"}


def _get_configured_key() -> str | None:
    key = os.environ.get("CHATTERBOX_API_KEY", "").strip()
    return key or None


def _extract_key(request: Request) -> str | None:
    header = request.headers.get("x-api-key")
    if header:
        return header.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    # WebSocket / query-string fallback (WS handshake can't always set headers)
    qp = request.query_params.get("api_key") if hasattr(request, "query_params") else None
    return qp.strip() if qp else None


def check_ws_key(websocket) -> bool:
    """Sync check for WebSocket connections. Returns True if allowed."""
    configured = _get_configured_key()
    if not configured:
        return True
    presented = (
        websocket.headers.get("x-api-key")
        or websocket.query_params.get("api_key")
        or ""
    ).strip()
    return bool(presented) and hmac.compare_digest(presented, configured)


async def api_key_middleware(request: Request, call_next: Callable):
    configured = _get_configured_key()
    if not configured:
        return await call_next(request)

    path = request.url.path
    if path in PUBLIC_PATHS or path.startswith("/static/"):
        return await call_next(request)

    # WebSocket handshakes come through the middleware as scope['type']=='http'
    # upgrading; we still let them through here and validate inside the handler
    # via check_ws_key() so we can close cleanly with a code.
    if request.headers.get("upgrade", "").lower() == "websocket":
        return await call_next(request)

    presented = _extract_key(request)
    if not presented or not hmac.compare_digest(presented, configured):
        return JSONResponse(
            {"error": "unauthorized", "hint": "set X-API-Key header"},
            status_code=401,
        )
    return await call_next(request)
