"""Security middleware for the SAM3 Inference Server."""
from typing import Optional, List
import time
import secrets

from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.config import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses."""
    
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_log = {}  # Simple in-memory storage (use Redis in production)
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP (consider X-Forwarded-For in production)
        client_ip = request.client.host
        
        # Get current minute
        current_minute = int(time.time() // 60)
        
        # Initialize client log if not exists
        if client_ip not in self.requests_log:
            self.requests_log[client_ip] = {}
        
        # Clean old entries
        client_log = self.requests_log[client_ip]
        for minute in list(client_log.keys()):
            if minute < current_minute - 1:
                del client_log[minute]
        
        # Increment request count for current minute
        client_log[current_minute] = client_log.get(current_minute, 0) + 1
        
        # Check if limit exceeded
        total_requests = sum(client_log.values())
        if total_requests > self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
        
        response = await call_next(request)
        return response


class APIKeyAuth:
    """API key authentication scheme."""
    
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.scheme = HTTPBearer(auto_error=False)
    
    async def __call__(self, request: Request) -> Optional[str]:
        if not settings.require_api_key:
            return None  # Skip auth if not required
            
        # Get credentials from header
        credentials: HTTPAuthorizationCredentials = await self.scheme(request)
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="API key required"
            )
        
        # Validate API key
        if credentials.credentials not in self.api_keys:
            raise HTTPException(
                status_code=403,
                detail="Invalid API key"
            )
        
        return credentials.credentials


# Create API key authenticator instance
api_key_auth = APIKeyAuth(settings.api_keys)