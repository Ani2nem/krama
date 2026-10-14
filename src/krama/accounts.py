"""Shops, users, and the keys to each board.

The platform grows tenants: a SHOP is a business running its own board, and a
shop is exactly one workspace - the isolation boundary the architecture already
had. Users carry an email, a password, and a role:

- ``admin``    - the platform operator. Sees every shop, every number.
- ``owner``    - a shop's principal. Their shop only; sees the money; may invite
                 more accounts into their own shop (the "three accounts under
                 him" case is just three users with his shop_id).
- ``dispatcher`` - a shop's day-to-day seat. Their shop only; money stripped,
                 exactly as the PIN split always worked.

Everything here is standard library on purpose. Passwords are scrypt with a
per-user salt; sessions are stateless HMAC-signed tokens (a Lambda between two
requests shares no memory, so the token must carry its own proof); invites are
random links with an expiry, shown to the admin to hand over - when an email
service is configured one day, the same link gets sent instead of copied.

The store is one JSON file (or S3 object via the workspace machinery later),
written atomically. Accounts change a few times a MONTH - this is config, not
an event stream, and a second storage engine would be pure ceremony.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Role = Literal["admin", "owner", "dispatcher"]

#: Sessions outlive a working day but not a lost laptop's week.
TOKEN_TTL_SECONDS = 36 * 3600
#: An invite that sat unopened for a week was lost, not pending.
INVITE_TTL_SECONDS = 7 * 24 * 3600

_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


class AccountsError(ValueError):
    """A request the account store refuses, with a reason a person can read."""


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return digest.hex()


@dataclass
class User:
    email: str
    role: Role
    shop_id: str  # "" for platform admins - they belong to no single shop
    salt: str
    password_hash: str
    active: bool = True

    def check(self, password: str) -> bool:
        expected = _hash_password(password, bytes.fromhex(self.salt))
        return hmac.compare_digest(expected, self.password_hash)


@dataclass
class Shop:
    shop_id: str
    name: str
    created_at: float


@dataclass
class Invite:
    token: str
    email: str
    shop_id: str
    role: Role
    expires_at: float


@dataclass
class AccountStore:
    """All shops, users and invites, behind one atomically-written file."""

    path: Path
    secret: str = ""
    shops: dict[str, Shop] = field(default_factory=dict)
    users: dict[str, User] = field(default_factory=dict)
    invites: dict[str, Invite] = field(default_factory=dict)

    # ------------------------------------------------------------------- store
    @classmethod
    def load(cls, path: Path) -> AccountStore:
        store = cls(path=path)
        if path.exists():
            raw = json.loads(path.read_text())
            store.secret = raw.get("secret", "")
            store.shops = {s["shop_id"]: Shop(**s) for s in raw.get("shops", [])}
            store.users = {u["email"]: User(**u) for u in raw.get("users", [])}
            store.invites = {i["token"]: Invite(**i) for i in raw.get("invites", [])}
        if not store.secret:
            # The signing secret lives WITH the accounts: whoever can read one can
            # read the other, and a missing env var must never silently mint a
            # fresh secret that logs every session out.
            store.secret = secrets.token_hex(32)
            store.save()
        return store

    def save(self) -> None:
        payload: dict[str, Any] = {
            "secret": self.secret,
            "shops": [vars(s) for s in self.shops.values()],
            "users": [vars(u) for u in self.users.values()],
            "invites": [vars(i) for i in self.invites.values()],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)

    # ------------------------------------------------------------------- shops
    def create_shop(self, name: str) -> Shop:
        cleaned = name.strip()
        if not cleaned:
            raise AccountsError("a shop needs a name")
        base = "".join(ch if ch.isalnum() else "-" for ch in cleaned.lower()).strip("-")
        shop_id = base or "shop"
        suffix = 2
        while shop_id in self.shops:
            shop_id = f"{base}-{suffix}"
            suffix += 1
        shop = Shop(shop_id=shop_id, name=cleaned, created_at=time.time())
        self.shops[shop_id] = shop
        self.save()
        return shop

    # ------------------------------------------------------------------- users
    def create_user(self, email: str, password: str, role: Role, shop_id: str) -> User:
        cleaned = email.strip().lower()
        if "@" not in cleaned:
            raise AccountsError("that does not look like an email address")
        if cleaned in self.users:
            raise AccountsError("an account with that email already exists")
        if role != "admin" and shop_id not in self.shops:
            raise AccountsError("no such shop")
        if len(password) < 8:
            raise AccountsError("the password needs at least 8 characters")
        salt = secrets.token_bytes(16)
        user = User(
            email=cleaned,
            role=role,
            shop_id="" if role == "admin" else shop_id,
            salt=salt.hex(),
            password_hash=_hash_password(password, salt),
        )
        self.users[cleaned] = user
        self.save()
        return user

    def authenticate(self, email: str, password: str) -> User | None:
        user = self.users.get(email.strip().lower())
        if user is None or not user.active:
            return None
        return user if user.check(password) else None

    def set_active(self, email: str, active: bool) -> None:
        user = self.users.get(email.strip().lower())
        if user is None:
            raise AccountsError("no such account")
        user.active = active
        self.save()

    # ----------------------------------------------------------------- invites
    def invite(self, email: str, shop_id: str, role: Role) -> Invite:
        cleaned = email.strip().lower()
        if "@" not in cleaned:
            raise AccountsError("that does not look like an email address")
        if cleaned in self.users:
            raise AccountsError("an account with that email already exists")
        if role == "admin":
            raise AccountsError("admins are created directly, never by invite link")
        if shop_id not in self.shops:
            raise AccountsError("no such shop")
        invite = Invite(
            token=secrets.token_urlsafe(24),
            email=cleaned,
            shop_id=shop_id,
            role=role,
            expires_at=time.time() + INVITE_TTL_SECONDS,
        )
        self.invites[invite.token] = invite
        self.save()
        return invite

    def accept_invite(self, token: str, password: str) -> User:
        invite = self.invites.get(token)
        if invite is None or invite.expires_at < time.time():
            raise AccountsError("that invite link is no longer valid - ask for a new one")
        user = self.create_user(invite.email, password, invite.role, invite.shop_id)
        del self.invites[token]
        self.save()
        return user

    # ------------------------------------------------------------------ tokens
    def issue_token(self, user: User) -> str:
        body = json.dumps(
            {"email": user.email, "exp": int(time.time()) + TOKEN_TTL_SECONDS},
            separators=(",", ":"),
        ).encode()
        sig = hmac.new(self.secret.encode(), body, hashlib.sha256).digest()
        encoded = base64.urlsafe_b64encode(body).decode()
        return encoded + "." + base64.urlsafe_b64encode(sig).decode()

    def verify_token(self, token: str) -> User | None:
        try:
            body_b64, sig_b64 = token.split(".", 1)
            body = base64.urlsafe_b64decode(body_b64)
            sig = base64.urlsafe_b64decode(sig_b64)
        except (ValueError, TypeError):
            return None
        expected = hmac.new(self.secret.encode(), body, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        claims = json.loads(body)
        if claims.get("exp", 0) < time.time():
            return None
        user = self.users.get(claims.get("email", ""))
        return user if user is not None and user.active else None
