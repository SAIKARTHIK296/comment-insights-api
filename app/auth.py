import hmac
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)


def get_api_token() -> str:
    return os.getenv("API_TOKEN", "").strip()


def validate_auth_config() -> None:
    if not get_api_token():
        raise RuntimeError("API_TOKEN environment variable is required to start the service.")


def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    expected = get_api_token()
    if not expected or credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
        )
    if not hmac.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
        )
    return credentials.credentials
