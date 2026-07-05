from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone

try:
    import bcrypt
except ModuleNotFoundError:  # pragma: no cover - exercised only when the optional wheel is absent
    bcrypt = None
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.base import utcnow
from app.db.models import User
from app.schemas.auth import TokenResponse, UserLogin, UserRegister, UserResponse


class AuthService:
    def __init__(self, session: AsyncSession, settings: Settings):
        self.session = session
        self.settings = settings

    async def register(self, payload: UserRegister) -> TokenResponse:
        if not self.settings.auth_allow_signup:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Signup is disabled"
            )
        email = _normalize_email(payload.email)
        existing = await self.user_by_email(email)
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Email is already registered"
            )
        user = User(email=email, password_hash=hash_password(payload.password))
        self.session.add(user)
        await self.session.commit()
        return self._token_response(user)

    async def login(self, payload: UserLogin) -> TokenResponse:
        user = await self.user_by_email(_normalize_email(payload.email))
        if user is None or not user.is_active or not verify_password(
            payload.password, user.password_hash
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
            )
        return self._token_response(user)

    async def user_by_email(self, email: str) -> User | None:
        return await self.session.scalar(select(User).where(User.email == email))

    async def user_by_id(self, user_id: str) -> User | None:
        return await self.session.scalar(select(User).where(User.id == user_id))

    def _token_response(self, user: User) -> TokenResponse:
        token, expires_at = create_access_token(
            user.id, self.settings.auth_secret_key, self.settings.auth_token_ttl_minutes
        )
        return TokenResponse(
            access_token=token,
            expires_at=expires_at,
            user=UserResponse.model_validate(user),
        )


def hash_password(password: str) -> str:
    if bcrypt is not None:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 600_000)
    return f"pbkdf2_sha256${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, password_hash: str) -> bool:
    if password_hash.startswith("pbkdf2_sha256$"):
        try:
            _, salt_b64, digest_b64 = password_hash.split("$", 2)
            salt = _b64decode(salt_b64)
            expected = _b64decode(digest_b64)
        except ValueError:
            return False
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 600_000)
        return hmac.compare_digest(candidate, expected)
    if bcrypt is None:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(
    user_id: str, secret_key: str, ttl_minutes: int
) -> tuple[str, datetime]:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=ttl_minutes)
    payload = {"sub": user_id, "exp": int(expires_at.timestamp())}
    payload_b64 = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _sign(payload_b64, secret_key)
    return f"{payload_b64}.{signature}", expires_at.replace(tzinfo=None)


def decode_access_token(token: str, secret_key: str) -> str:
    try:
        payload_b64, signature = token.split(".", 1)
    except ValueError as exc:
        raise _invalid_token() from exc
    expected = _sign(payload_b64, secret_key)
    if not hmac.compare_digest(signature, expected):
        raise _invalid_token()
    try:
        payload = json.loads(_b64decode(payload_b64))
        user_id = payload["sub"]
        expires_at = int(payload["exp"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _invalid_token() from exc
    if not isinstance(user_id, str) or not user_id:
        raise _invalid_token()
    if utcnow().timestamp() >= expires_at:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    return user_id


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _sign(payload_b64: str, secret_key: str) -> str:
    digest = hmac.new(
        secret_key.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256
    ).digest()
    return _b64encode(digest)


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _invalid_token() -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
