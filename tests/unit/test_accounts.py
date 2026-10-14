"""Shops, users, and who may see which board.

The platform's isolation promise in executable form: a shop is its own
workspace, a role is the credential, and single-shop mode (auth off) behaves
exactly as every other test in this suite assumes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def portal(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("KRAMA_AUTH", "on")
    monkeypatch.setenv("KRAMA_SHOPS", str(tmp_path / "shops"))
    monkeypatch.setenv("KRAMA_ADMIN_EMAIL", "ani@krama.app")
    monkeypatch.setenv("KRAMA_ADMIN_PASSWORD", "let-me-in-1")
    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    monkeypatch.delenv("KRAMA_API_KEY", raising=False)

    from krama.api.main import app

    return TestClient(app)


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    reply = client.post("/api/auth/login", json={"email": email, "password": password})
    assert reply.status_code == 200, reply.text
    body: dict[str, str] = reply.json()
    return body


def test_the_portal_lifecycle(portal: TestClient):
    """Bootstrap the admin, open a shop, invite its owner, the owner grows their
    own team, and a dispatcher's board hides the money - end to end."""
    # Signed out means signed out.
    assert portal.get("/api/world").status_code == 401

    # The first login mints the platform admin from the environment, once.
    admin = _login(portal, "ani@krama.app", "let-me-in-1")
    assert admin["role"] == "admin"
    auth_admin = {"Authorization": f"Bearer {admin['token']}"}

    # A shop is born with a working board.
    shop = portal.post("/api/admin/shops", headers=auth_admin, json={"name": "Glass Guru"}).json()
    assert shop["shop_id"] == "glass-guru"

    # The admin invites the shop's principal; the link is handed over by hand.
    invite = portal.post(
        "/api/admin/invites",
        headers=auth_admin,
        json={"email": "guru@example.com", "shop_id": "glass-guru", "role": "owner"},
    ).json()
    token = invite["link"].split("=", 1)[1]

    joined = portal.post("/api/auth/join", json={"token": token, "password": "gl4ss-guru"})
    assert joined.status_code == 200
    owner = _login(portal, "guru@example.com", "gl4ss-guru")
    assert owner["role"] == "owner" and owner["shop_id"] == "glass-guru"
    auth_owner = {"Authorization": f"Bearer {owner['token']}"}

    # The owner sees their own board, money included.
    world = portal.get("/api/world", headers=auth_owner)
    assert world.status_code == 200

    # "Three accounts under him": the owner invites their own dispatcher...
    second = portal.post(
        "/api/shop/invites",
        headers=auth_owner,
        json={"email": "desk@example.com", "role": "dispatcher"},
    ).json()
    desk_token = second["link"].split("=", 1)[1]
    portal.post("/api/auth/join", json={"token": desk_token, "password": "desk-pass-1"})
    desk = _login(portal, "desk@example.com", "desk-pass-1")
    assert desk["shop_id"] == "glass-guru" and desk["role"] == "dispatcher"

    # ...and the dispatcher's board is money-stripped, exactly like the PIN split.
    auth_desk = {"Authorization": f"Bearer {desk['token']}"}
    pricing = portal.get("/api/config/pricing", headers=auth_desk)
    assert pricing.status_code == 403, "the rate card is the owner's"

    # A dispatcher cannot run the admin surface, or grow the team.
    assert portal.get("/api/admin/shops", headers=auth_desk).status_code == 403
    assert (
        portal.post(
            "/api/shop/invites",
            headers=auth_desk,
            json={"email": "x@example.com"},
        ).status_code
        == 403
    )

    # Deactivation locks the door mid-session.
    portal.post(
        "/api/admin/users/desk@example.com/active",
        headers=auth_admin,
        params={"active": "false"},
    )
    assert portal.get("/api/world", headers=auth_desk).status_code == 401


def test_every_shop_is_its_own_world(portal: TestClient):
    """The isolation promise: a booking in one shop never shows in another, and
    the admin walks between boards with a header."""
    admin = _login(portal, "ani@krama.app", "let-me-in-1")
    auth = {"Authorization": f"Bearer {admin['token']}"}
    portal.post("/api/admin/shops", headers=auth, json={"name": "North Shop"})
    portal.post("/api/admin/shops", headers=auth, json={"name": "South Shop"})

    # The admin must say which board they mean.
    assert portal.get("/api/world", headers=auth).status_code == 400

    north = {**auth, "X-Shop": "north-shop"}
    south = {**auth, "X-Shop": "south-shop"}
    draft = {
        "customer_name": "Northsider",
        "phone": "9990001111",
        "address": "north st",
        "service_type": "residential_window_replacement",
        "duration_minutes": 60,
        "duration_confidence": 30,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    today = datetime.now(ZoneInfo("America/Chicago")).date()
    bump = (7 - today.weekday()) % 7 or 7
    monday = (today + timedelta(days=bump)).isoformat()
    booked = portal.post(
        "/api/book", headers=north, json={"draft": draft, "date": monday, "arrival": "09:00"}
    )
    assert booked.status_code == 200, booked.text

    north_world = portal.get("/api/world", headers=north).json()
    south_world = portal.get("/api/world", headers=south).json()
    north_jobs = {j["customer_name"] for j in north_world["jobs"]}
    south_jobs = {j["customer_name"] for j in south_world["jobs"]}
    assert "Northsider" in north_jobs
    assert "Northsider" not in south_jobs, "a booking never leaks across shops"


def test_invites_expire_and_cannot_be_reused(tmp_path):
    from krama.accounts import AccountsError, AccountStore

    store = AccountStore.load(tmp_path / "accounts.json")
    shop = store.create_shop("Testers")
    invite = store.invite("one@example.com", shop.shop_id, "dispatcher")
    store.accept_invite(invite.token, "a-password-1")
    with pytest.raises(AccountsError):
        store.accept_invite(invite.token, "a-password-1")

    stale = store.invite("two@example.com", shop.shop_id, "dispatcher")
    store.invites[stale.token].expires_at = 0
    with pytest.raises(AccountsError):
        store.accept_invite(stale.token, "a-password-1")
