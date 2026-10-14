import { useEffect, useState } from "react";

import { api, ApiError, currentAdminShop, setAdminShop } from "../api";

/**
 * The doors of the building: sign in, join by invite, and the admin's room.
 *
 * Multi-tenant mode only - a laptop board (auth off) never renders any of this.
 * The grammar matches the rest of the product: a shop is a business running its
 * own board; an invite is a LINK the admin hands over (email delivery is a
 * configuration away, and a link you watched someone receive beats a silent
 * delivery failure); who you are decides what you see.
 */

export function LoginCard({
  onSignedIn,
}: {
  onSignedIn: (who: {
    token: string;
    email: string;
    role: string;
    shop_id: string;
    shop_name: string;
  }) => void;
}) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit() {
    setBusy(true);
    setError("");
    try {
      const who = await api.login(email, password);
      onSignedIn(who);
    } catch (exc) {
      setError((exc as ApiError).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="gate">
      <div className="gate__card">
        <h1>Krama</h1>
        <p className="muted">Sign in to your shop's board.</p>
        <input
          type="email"
          placeholder="email"
          value={email}
          autoFocus
          onChange={(e) => setEmail(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void submit()}
        />
        <input
          type="password"
          placeholder="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void submit()}
        />
        <button className="primary" disabled={busy || !email || !password} onClick={() => void submit()}>
          {busy ? "Checking…" : "Sign in"}
        </button>
        {error && <p className="gate__error">{error}</p>}
      </div>
    </div>
  );
}

export function JoinCard({ token, onJoined }: { token: string; onJoined: () => void }) {
  const [password, setPassword] = useState("");
  const [again, setAgain] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);

  async function submit() {
    setBusy(true);
    setError("");
    try {
      await api.join(token, password);
      setDone(true);
    } catch (exc) {
      setError((exc as ApiError).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="gate">
      <div className="gate__card">
        <h1>Krama</h1>
        {done ? (
          <>
            <p>Your account is ready.</p>
            <button className="primary" onClick={onJoined}>
              Go to sign in
            </button>
          </>
        ) : (
          <>
            <p className="muted">You were invited - choose a password to finish.</p>
            <input
              type="password"
              placeholder="password (8+ characters)"
              value={password}
              autoFocus
              onChange={(e) => setPassword(e.target.value)}
            />
            <input
              type="password"
              placeholder="same password again"
              value={again}
              onChange={(e) => setAgain(e.target.value)}
            />
            <button
              className="primary"
              disabled={busy || password.length < 8 || password !== again}
              title={password !== again ? "the two passwords differ" : ""}
              onClick={() => void submit()}
            >
              {busy ? "Creating…" : "Create my account"}
            </button>
            {error && <p className="gate__error">{error}</p>}
          </>
        )}
      </div>
    </div>
  );
}

export function AdminPanel({
  onClose,
  onShopChosen,
}: {
  onClose: () => void;
  /** The admin picked a shop to WORK as - the whole board re-reads as that shop. */
  onShopChosen: () => void;
}) {
  const [shops, setShops] = useState<{ shop_id: string; name: string; users: number }[]>([]);
  const [chosen, setChosen] = useState(currentAdminShop());
  const [people, setPeople] = useState<{
    users: { email: string; role: string; active: boolean }[];
    invites: { email: string; role: string; link: string }[];
  }>({ users: [], invites: [] });
  const [newShop, setNewShop] = useState("");
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState("dispatcher");
  const [freshLink, setFreshLink] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");

  async function refreshShops() {
    setShops((await api.shops()).shops);
  }
  async function refreshPeople(shopId: string) {
    if (!shopId) return;
    const reply = await api.shopUsers(shopId);
    setPeople({ users: reply.users, invites: reply.invites });
  }
  useEffect(() => {
    void refreshShops();
    if (chosen) void refreshPeople(chosen);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function run(work: () => Promise<unknown>, done: string) {
    setBusy(true);
    setStatus("");
    try {
      await work();
      setStatus(done);
    } catch (exc) {
      setStatus((exc as ApiError).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="config"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="config__sheet">
        <div className="config__head">
          <h2>Shops and accounts</h2>
          <button onClick={onClose}>Close</button>
        </div>

        <h3 className="config__vanshead">Shops</h3>
        <p className="config__hint">
          Each shop is its own board - its own bookings, crew and money. Pick one to
          work as; everything behind this sheet becomes that shop.
        </p>
        <div className="portal__shops">
          {shops.map((shop) => (
            <button
              key={shop.shop_id}
              className={`portal__shop${chosen === shop.shop_id ? " on" : ""}`}
              disabled={busy}
              onClick={() => {
                setChosen(shop.shop_id);
                setAdminShop(shop.shop_id);
                setFreshLink("");
                void refreshPeople(shop.shop_id);
                onShopChosen();
              }}
            >
              <strong>{shop.name}</strong>
              <span className="muted"> · {shop.users} account{shop.users === 1 ? "" : "s"}</span>
            </button>
          ))}
          <span className="portal__newshop">
            <input
              placeholder="new shop name"
              value={newShop}
              disabled={busy}
              onChange={(e) => setNewShop(e.target.value)}
            />
            <button
              disabled={busy || !newShop.trim()}
              onClick={() =>
                void run(async () => {
                  await api.createShop(newShop.trim());
                  setNewShop("");
                  await refreshShops();
                }, "shop created, board ready")
              }
            >
              Open a shop
            </button>
          </span>
        </div>

        {chosen && (
          <>
            <h3 className="config__vanshead">
              Accounts at {shops.find((s) => s.shop_id === chosen)?.name ?? chosen}
            </h3>
            <div className="portal__people">
              {people.users.map((u) => (
                <div key={u.email} className="portal__person">
                  <span>
                    {u.email} <em className="muted">· {u.role}</em>
                    {!u.active && <em className="portal__off"> · deactivated</em>}
                  </span>
                  <button
                    className={u.active ? "danger" : ""}
                    disabled={busy}
                    onClick={() =>
                      void run(async () => {
                        await api.setUserActive(u.email, !u.active);
                        await refreshPeople(chosen);
                      }, u.active ? "signed out and locked" : "reactivated")
                    }
                  >
                    {u.active ? "Deactivate" : "Reactivate"}
                  </button>
                </div>
              ))}
              {people.invites.map((i) => (
                <div key={i.link} className="portal__person">
                  <span>
                    {i.email} <em className="muted">· invited as {i.role}</em>
                  </span>
                  <button
                    disabled={busy}
                    onClick={() => {
                      void navigator.clipboard.writeText(
                        `${window.location.origin}/${i.link}`,
                      );
                      setStatus(`invite link for ${i.email} copied - send it to them`);
                    }}
                  >
                    Copy invite link
                  </button>
                </div>
              ))}
              <div className="portal__invite">
                <input
                  placeholder="email to invite"
                  value={inviteEmail}
                  disabled={busy}
                  onChange={(e) => setInviteEmail(e.target.value)}
                />
                <select
                  value={inviteRole}
                  disabled={busy}
                  onChange={(e) => setInviteRole(e.target.value)}
                >
                  <option value="dispatcher">dispatcher (no money)</option>
                  <option value="owner">owner (sees everything)</option>
                </select>
                <button
                  disabled={busy || !inviteEmail.includes("@")}
                  onClick={() =>
                    void run(async () => {
                      const made = await api.adminInvite(inviteEmail, chosen, inviteRole);
                      setFreshLink(`${window.location.origin}/${made.link}`);
                      setInviteEmail("");
                      await refreshPeople(chosen);
                    }, "invite minted - copy the link below and send it")
                  }
                >
                  Invite
                </button>
              </div>
              {freshLink && (
                <p className="portal__link">
                  <code>{freshLink}</code>{" "}
                  <button onClick={() => void navigator.clipboard.writeText(freshLink)}>
                    Copy
                  </button>
                </p>
              )}
            </div>
          </>
        )}
        {status && <p className="config__status">{status}</p>}
      </div>
    </div>
  );
}
