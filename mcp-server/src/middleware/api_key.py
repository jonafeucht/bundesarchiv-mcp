import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        API_KEY = os.environ.get("MCP_API_KEY")
        USE_API_KEY = os.environ.get("USE_API_KEY", "true").lower() == "true"

        if not USE_API_KEY:
            return await call_next(request)
        key = request.headers.get("x-api-key") or request.query_params.get("api_key")
        if key != API_KEY:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)
