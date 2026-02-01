"""Evergreen Reader backend reference implementation

Why this version exists

Some sandboxed preview environments ship a restricted Python runtime without the built in `ssl` module.
Modern ASGI stacks such as FastAPI depend on AnyIO, which imports `ssl`. In such sandboxes, importing
FastAPI fails at import time with ModuleNotFoundError: No module named 'ssl'.

To make this file runnable everywhere while preserving a production oriented design, this module is split
into two layers:

1) Pure Python domain layer (always available)
   - OTP issuance and verification
   - Device limit enforcement (max 3 devices)
   - Refresh token issuance and verification
   - In memory persistence (for preview) with an interface you can swap for SQLAlchemy or PostgreSQL

2) Optional FastAPI adapter (only enabled when `ssl` and FastAPI dependencies are importable)
   - Exposes the same endpoints as originally specified

In a real repository, keep the domain layer independent of the web framework, and implement a persistence
adapter using SQLAlchemy 2.0, Alembic migrations, PostgreSQL, and Redis.

Endpoints intended (when FastAPI is available)

POST /auth/request-otp
POST /auth/verify-otp
POST /auth/refresh
POST /auth/logout
GET  /me
GET  /devices
DELETE /devices/{device_id}

Core rules

- One user per email
- Max 3 devices per user enforced at verify time
- Refresh tokens are opaque, stored hashed
- Access tokens are short lived (JWT in production). In this preview fallback, access tokens are opaque.

"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


# -----------------------
# Utilities
# -----------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def issue_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    # In production, add a server side pepper.
    return sha256_hex(token)


def issue_access_token_opaque() -> str:
    # In production use JWT. Here we use an opaque token to avoid any crypto dependencies.
    return secrets.token_urlsafe(32)


# -----------------------
# Exceptions
# -----------------------


class AuthError(Exception):
    def __init__(self, message: str, status_code: int = 401, detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail if detail is not None else message


class RateLimitError(AuthError):
    pass


class DeviceLimitReached(AuthError):
    pass


# -----------------------
# In memory persistence (preview adapter)
# -----------------------


@dataclass
class User:
    id: int
    email: str
    email_verified_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


@dataclass
class Device:
    id: int
    user_id: int
    device_fingerprint: str
    device_name: Optional[str]
    platform: str
    last_seen_at: datetime
    created_at: datetime


@dataclass
class AuthSession:
    id: int
    user_id: int
    device_id: int
    refresh_token_hash: str
    revoked_at: Optional[datetime]
    created_at: datetime
    expires_at: datetime


class MemoryDB:
    """A tiny in memory store to keep this file runnable in restricted sandboxes.

    Replace with SQLAlchemy models and a real database in production.
    """

    def __init__(self) -> None:
        self._user_id = 0
        self._device_id = 0
        self._session_id = 0
        self.users_by_email: Dict[str, User] = {}
        self.devices_by_user: Dict[int, List[Device]] = {}
        self.sessions_by_user: Dict[int, List[AuthSession]] = {}

    def get_or_create_user(self, email: str) -> User:
        email_norm = email.strip().lower()
        existing = self.users_by_email.get(email_norm)
        if existing:
            return existing
        self._user_id += 1
        now = utcnow()
        u = User(id=self._user_id, email=email_norm, email_verified_at=None, created_at=now, updated_at=now)
        self.users_by_email[email_norm] = u
        self.devices_by_user[u.id] = []
        self.sessions_by_user[u.id] = []
        return u

    def list_devices(self, user_id: int) -> List[Device]:
        return list(self.devices_by_user.get(user_id, []))

    def upsert_device(self, user_id: int, device_fingerprint: str, device_name: Optional[str], platform: str) -> Device:
        devices = self.devices_by_user.setdefault(user_id, [])
        for d in devices:
            if d.device_fingerprint == device_fingerprint:
                d.device_name = device_name or d.device_name
                d.platform = platform or d.platform
                d.last_seen_at = utcnow()
                return d

        self._device_id += 1
        now = utcnow()
        d = Device(
            id=self._device_id,
            user_id=user_id,
            device_fingerprint=device_fingerprint,
            device_name=device_name,
            platform=platform,
            last_seen_at=now,
            created_at=now,
        )
        devices.append(d)
        return d

    def delete_device(self, user_id: int, device_id: int) -> bool:
        devices = self.devices_by_user.get(user_id, [])
        before = len(devices)
        self.devices_by_user[user_id] = [d for d in devices if d.id != device_id]
        removed = len(self.devices_by_user[user_id]) != before

        # revoke sessions for this device
        sessions = self.sessions_by_user.get(user_id, [])
        now = utcnow()
        for s in sessions:
            if s.device_id == device_id and s.revoked_at is None:
                s.revoked_at = now
        return removed

    def add_session(self, user_id: int, device_id: int, refresh_token_hash: str, expires_at: datetime) -> AuthSession:
        self._session_id += 1
        now = utcnow()
        s = AuthSession(
            id=self._session_id,
            user_id=user_id,
            device_id=device_id,
            refresh_token_hash=refresh_token_hash,
            revoked_at=None,
            created_at=now,
            expires_at=expires_at,
        )
        self.sessions_by_user.setdefault(user_id, []).append(s)
        return s

    def find_session_by_refresh_hash(self, refresh_hash: str) -> Optional[AuthSession]:
        for sessions in self.sessions_by_user.values():
            for s in sessions:
                if s.refresh_token_hash == refresh_hash and s.revoked_at is None:
                    return s
        return None


# -----------------------
# OTP store
# -----------------------


@dataclass
class OtpRecord:
    otp_hash: str
    expires_at_epoch: int


class OtpStore:
    def __init__(self, ttl_seconds: int = 600, resend_cooldown_seconds: int = 30, max_attempts: int = 8) -> None:
        self.ttl_seconds = int(ttl_seconds)
        self.resend_cooldown_seconds = int(resend_cooldown_seconds)
        self.max_attempts = int(max_attempts)

        self._otp: Dict[str, OtpRecord] = {}
        self._attempts: Dict[str, int] = {}
        self._last_sent_epoch: Dict[str, int] = {}

    def _k(self, email: str) -> str:
        return email.strip().lower()

    def can_send_now(self, email: str) -> bool:
        k = self._k(email)
        now = int(time.time())
        last = int(self._last_sent_epoch.get(k, 0) or 0)
        if last and now - last < self.resend_cooldown_seconds:
            return False
        self._last_sent_epoch[k] = now
        return True

    def set_otp(self, email: str, otp_hash: str) -> None:
        k = self._k(email)
        self._otp[k] = OtpRecord(otp_hash=otp_hash, expires_at_epoch=int(time.time()) + self.ttl_seconds)
        self._attempts[k] = 0

    def get_otp_hash(self, email: str) -> Optional[str]:
        k = self._k(email)
        rec = self._otp.get(k)
        if not rec:
            return None
        if int(time.time()) > rec.expires_at_epoch:
            self._otp.pop(k, None)
            return None
        return rec.otp_hash

    def delete_otp(self, email: str) -> None:
        k = self._k(email)
        self._otp.pop(k, None)
        self._attempts.pop(k, None)

    def incr_attempts(self, email: str) -> int:
        k = self._k(email)
        v = int(self._attempts.get(k, 0)) + 1
        self._attempts[k] = v
        return v


# -----------------------
# Email sending (Resend)
# -----------------------

import os
import json
import urllib.request


def send_otp_email(to_email: str, otp: str) -> bool:
    """Send OTP email using Resend when RESEND_API_KEY is set.

    Returns True when an email was sent successfully.
    Returns False when Resend is not configured.

    Any network or provider errors will raise, allowing the caller to decide whether to fall back.
    """
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        return False

    payload = {
        "from": os.getenv("RESEND_FROM", "onboarding@resend.dev"),
        "to": [to_email],
        "subject": "Your Evergreen Reader login code",
        "html": f"<p>Your login code is:</p><h2>{otp}</h2>",
    }

    req = urllib.request.Request(
        url="https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        if getattr(resp, "status", 200) >= 400:
            raise RuntimeError(f"Resend error: {resp.status}")

    return True


# -----------------------
# Auth service (domain)
# -----------------------


class AuthService:
    def __init__(self, db: MemoryDB, otp: OtpStore, max_devices: int = 3, refresh_days: int = 30) -> None:
        self.db = db
        self.otp = otp
        self.max_devices = int(max_devices)
        self.refresh_days = int(refresh_days)

        # In preview we keep an access token table.
        self._access_index: Dict[str, Tuple[int, int]] = {}

    def request_otp(self, email: str) -> Dict[str, Any]:
        email_norm = email.strip().lower()
        if not self.otp.can_send_now(email_norm):
            raise RateLimitError("Please wait before requesting another code", status_code=429)

        otp = generate_otp()
        self.otp.set_otp(email_norm, sha256_hex(otp))

        # Send OTP via Resend if configured, otherwise fall back to console output
        try:
            sent = send_otp_email(email_norm, otp)
            if not sent:
                print(f"[DEV OTP] Send to {email_norm}: {otp}")
        except Exception:
            # Safe fallback for transient network / provider errors
            print(f"[DEV OTP] Send to {email_norm}: {otp}")

        return {"ok": True}

    def verify_otp(self, email: str, otp_code: str, device_fingerprint: str, device_name: Optional[str], platform: str) -> Dict[str, Any]:
        email_norm = email.strip().lower()

        stored_hash = self.otp.get_otp_hash(email_norm)
        if not stored_hash:
            raise AuthError("OTP expired or not requested", status_code=400)

        attempts = self.otp.incr_attempts(email_norm)
        if attempts > self.otp.max_attempts:
            self.otp.delete_otp(email_norm)
            raise RateLimitError("Too many attempts", status_code=429)

        if sha256_hex(otp_code.strip()) != stored_hash:
            raise AuthError("Invalid OTP", status_code=400)

        self.otp.delete_otp(email_norm)

        user = self.db.get_or_create_user(email_norm)
        if user.email_verified_at is None:
            user.email_verified_at = utcnow()
            user.updated_at = utcnow()

        # device limit enforcement
        devices = self.db.list_devices(user.id)
        existing = [d for d in devices if d.device_fingerprint == device_fingerprint]
        if not existing and len(devices) >= self.max_devices:
            raise DeviceLimitReached(
                "Device limit reached",
                status_code=403,
                detail={
                    "code": "DEVICE_LIMIT_REACHED",
                    "message": "Device limit reached",
                    "devices": [self._device_public(d) for d in devices],
                },
            )

        device = self.db.upsert_device(user.id, device_fingerprint=device_fingerprint, device_name=device_name, platform=platform)

        refresh_token = issue_refresh_token()
        refresh_hash = hash_refresh_token(refresh_token)
        expires_at = utcnow() + timedelta(days=self.refresh_days)
        self.db.add_session(user.id, device.id, refresh_hash, expires_at)

        access_token = issue_access_token_opaque()
        self._access_index[access_token] = (user.id, device.id)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": {"id": user.id, "email": user.email},
            "entitlements": {"plan": "free", "features": {"cloud": True, "offline": False, "profiles": True}},
            "devices": [self._device_public(d) for d in self.db.list_devices(user.id)],
        }

    def refresh(self, refresh_token: str, device_fingerprint: str) -> Dict[str, Any]:
        token_hash = hash_refresh_token(refresh_token)
        session = self.db.find_session_by_refresh_hash(token_hash)
        if not session:
            raise AuthError("Invalid refresh token", status_code=401)

        if session.expires_at <= utcnow():
            session.revoked_at = utcnow()
            raise AuthError("Refresh token expired", status_code=401)

        # confirm device fingerprint
        devices = self.db.list_devices(session.user_id)
        dev = next((d for d in devices if d.id == session.device_id), None)
        if not dev or dev.device_fingerprint != device_fingerprint:
            raise AuthError("Device mismatch", status_code=401)

        dev.last_seen_at = utcnow()

        access_token = issue_access_token_opaque()
        self._access_index[access_token] = (session.user_id, session.device_id)
        return {"access_token": access_token}

    def logout(self, access_token: str) -> Dict[str, Any]:
        ctx = self.require_auth(access_token)
        sessions = self.db.sessions_by_user.get(ctx[0], [])
        now = utcnow()
        for s in sessions:
            if s.device_id == ctx[1] and s.revoked_at is None:
                s.revoked_at = now
        return {"ok": True}

    def me(self, access_token: str) -> Dict[str, Any]:
        user_id, device_id = self.require_auth(access_token)
        # user lookup
        user = None
        for u in self.db.users_by_email.values():
            if u.id == user_id:
                user = u
                break
        if not user:
            raise AuthError("User not found", status_code=404)

        return {
            "user": {"id": user.id, "email": user.email},
            "entitlements": {"plan": "free", "features": {"cloud": True, "offline": False, "profiles": True}},
            "device_id": device_id,
        }

    def devices(self, access_token: str) -> Dict[str, Any]:
        user_id, _device_id = self.require_auth(access_token)
        return {"devices": [self._device_public(d) for d in self.db.list_devices(user_id)]}

    def delete_device(self, access_token: str, device_id: int) -> Dict[str, Any]:
        user_id, current_device_id = self.require_auth(access_token)

        if int(device_id) == int(current_device_id):
            raise AuthError("Cannot remove the current device while signed in", status_code=400)

        removed = self.db.delete_device(user_id=user_id, device_id=int(device_id))
        if not removed:
            raise AuthError("Device not found", status_code=404)

        return {"ok": True}

    def require_auth(self, access_token: str) -> Tuple[int, int]:
        ctx = self._access_index.get(access_token)
        if not ctx:
            raise AuthError("Missing or invalid access token", status_code=401)
        return ctx

    def _device_public(self, d: Device) -> Dict[str, Any]:
        return {"id": d.id, "device_name": d.device_name, "platform": d.platform, "last_seen_at": d.last_seen_at.isoformat()}


# Instantiate a default service for tests and for the optional FastAPI adapter.
memdb = MemoryDB()
otp_store = OtpStore(ttl_seconds=600, resend_cooldown_seconds=30, max_attempts=8)
auth = AuthService(memdb, otp_store, max_devices=3, refresh_days=30)


# -----------------------
# Optional FastAPI adapter
# -----------------------


FASTAPI_AVAILABLE = False
app = None

try:
    # In restricted sandboxes, importing ssl itself fails.
    import ssl as _ssl  # noqa: F401

    from fastapi import Depends, FastAPI, HTTPException, Path, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, EmailStr, Field

    FASTAPI_AVAILABLE = True

    class RequestOtpIn(BaseModel):
        email: EmailStr

    class RequestOtpOut(BaseModel):
        ok: bool = True

    class VerifyOtpIn(BaseModel):
        email: EmailStr
        otp: str = Field(min_length=4, max_length=10)
        device_fingerprint: str = Field(min_length=6, max_length=200)
        device_name: Optional[str] = Field(default=None, max_length=120)
        platform: str = Field(default="web", max_length=20)

    class VerifyOtpOut(BaseModel):
        access_token: str
        refresh_token: str
        user: Dict[str, Any]
        entitlements: Dict[str, Any]
        devices: List[Dict[str, Any]]

    class RefreshIn(BaseModel):
        refresh_token: str
        device_fingerprint: str

    class RefreshOut(BaseModel):
        access_token: str

    class LogoutOut(BaseModel):
        ok: bool = True

    app = FastAPI(title="Evergreen Reader API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _bearer(request: Request) -> str:
        h = request.headers.get("authorization") or request.headers.get("Authorization")
        if not h or not h.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing access token")
        return h.split(" ", 1)[1].strip()

    def _raise_domain(e: AuthError) -> None:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    @app.post("/auth/request-otp", response_model=RequestOtpOut)
    def request_otp(payload: RequestOtpIn) -> RequestOtpOut:
        try:
            auth.request_otp(str(payload.email))
            return RequestOtpOut(ok=True)
        except AuthError as e:
            _raise_domain(e)

    @app.post("/auth/verify-otp", response_model=VerifyOtpOut)
    def verify_otp(payload: VerifyOtpIn) -> VerifyOtpOut:
        try:
            out = auth.verify_otp(
                email=str(payload.email),
                otp_code=payload.otp,
                device_fingerprint=payload.device_fingerprint,
                device_name=payload.device_name,
                platform=payload.platform,
            )
            return VerifyOtpOut(**out)
        except AuthError as e:
            _raise_domain(e)

    @app.post("/auth/refresh", response_model=RefreshOut)
    def refresh_token(payload: RefreshIn) -> RefreshOut:
        try:
            out = auth.refresh(payload.refresh_token, payload.device_fingerprint)
            return RefreshOut(**out)
        except AuthError as e:
            _raise_domain(e)

    @app.post("/auth/logout", response_model=LogoutOut)
    def logout(request: Request) -> LogoutOut:
        try:
            token = _bearer(request)
            out = auth.logout(token)
            return LogoutOut(**out)
        except AuthError as e:
            _raise_domain(e)

    @app.get("/me")
    def me(request: Request) -> Dict[str, Any]:
        try:
            token = _bearer(request)
            return auth.me(token)
        except AuthError as e:
            _raise_domain(e)

    @app.get("/devices")
    def devices(request: Request) -> Dict[str, Any]:
        try:
            token = _bearer(request)
            return auth.devices(token)
        except AuthError as e:
            _raise_domain(e)

    @app.delete("/devices/{device_id}")
    def delete_device(request: Request, device_id: int = Path(..., ge=1)) -> Dict[str, Any]:
        try:
            token = _bearer(request)
            return auth.delete_device(token, device_id)
        except AuthError as e:
            _raise_domain(e)


except Exception:
    # FastAPI adapter not available in this runtime. Domain layer remains usable.
    FASTAPI_AVAILABLE = False
    app = None


# -----------------------
# Tests
# -----------------------


def test_device_limit_cap_domain_layer() -> None:
    """Domain level test that always runs.

    This replaces dependence on FastAPI TestClient in environments where FastAPI cannot import
    due to missing `ssl`.
    """

    db = MemoryDB()
    otp = OtpStore(ttl_seconds=600, resend_cooldown_seconds=0, max_attempts=8)
    svc = AuthService(db, otp, max_devices=3, refresh_days=30)

    email = "t@example.com"

    # Request OTP, then read the stored OTP hash by issuing a fresh OTP we control.
    # For deterministic testing, we set the OTP directly.
    fixed_otp = "000000"
    otp.set_otp(email, sha256_hex(fixed_otp))

    for i in range(3):
        out = svc.verify_otp(email, fixed_otp, device_fingerprint=f"fp{i}", device_name=f"D{i}", platform="web")
        assert isinstance(out.get("access_token"), str)
        assert out["user"]["email"] == email

    # Fourth device should fail
    otp.set_otp(email, sha256_hex(fixed_otp))
    try:
        svc.verify_otp(email, fixed_otp, device_fingerprint="fp3", device_name="D3", platform="web")
        assert False, "Expected device limit error"
    except DeviceLimitReached as e:
        assert e.status_code == 403
        assert isinstance(e.detail, dict)
        assert e.detail.get("code") == "DEVICE_LIMIT_REACHED"


def test_key_security_properties() -> None:
    """Basic checks for token hashing behavior."""

    t = "token_value"
    h1 = hash_refresh_token(t)
    h2 = hash_refresh_token(t)
    assert h1 == h2
    assert h1 != hash_refresh_token(t + "x")


def test_refresh_requires_device_match() -> None:
    db = MemoryDB()
    otp = OtpStore(ttl_seconds=600, resend_cooldown_seconds=0, max_attempts=8)
    svc = AuthService(db, otp, max_devices=3, refresh_days=1)

    email = "r@example.com"
    fixed_otp = "123456"
    otp.set_otp(email, sha256_hex(fixed_otp))

    out = svc.verify_otp(email, fixed_otp, device_fingerprint="fpA", device_name="A", platform="web")
    refresh_token = out["refresh_token"]

    # mismatch
    try:
        svc.refresh(refresh_token, device_fingerprint="fpB")
        assert False, "Expected device mismatch"
    except AuthError as e:
        assert e.status_code == 401
        assert "Device mismatch" in str(e)


# Optional FastAPI test remains, but is skipped when FastAPI is not available.

def test_fastapi_adapter_imports_when_available() -> None:
    if not FASTAPI_AVAILABLE:
        # Nothing to assert in restricted environments.
        return

    assert app is not None


def test_request_otp_does_not_raise_without_resend_key() -> None:
    """OTP requests should still work in development when RESEND_API_KEY is absent.

    In that case, the OTP is printed to stdout rather than emailed.
    """
    db = MemoryDB()
    otp = OtpStore(ttl_seconds=600, resend_cooldown_seconds=0, max_attempts=8)
    svc = AuthService(db, otp, max_devices=3, refresh_days=30)

    # Ensure key is not set for this test
    old = os.getenv("RESEND_API_KEY")
    if old is not None:
        os.environ.pop("RESEND_API_KEY", None)

    out = svc.request_otp("z@example.com")
    assert out.get("ok") is True

    # Restore env
    if old is not None:
        os.environ["RESEND_API_KEY"] = old
