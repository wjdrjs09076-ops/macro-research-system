"""Quick Alpaca paper-trading connectivity check."""
from __future__ import annotations

import os
from pathlib import Path

import requests


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    load_env(Path(__file__).resolve().parent / ".env")

    base = os.environ["ALPACA_ENDPOINT"].rstrip("/")
    headers = {
        "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
        "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET"],
    }

    r = requests.get(f"{base}/account", headers=headers, timeout=15)
    print(f"GET /account -> {r.status_code}")
    if r.status_code != 200:
        print(r.text[:500])
        return

    a = r.json()
    print(f"  account_number : {a.get('account_number')}")
    print(f"  status         : {a.get('status')}")
    print(f"  currency       : {a.get('currency')}")
    print(f"  cash           : {a.get('cash')}")
    print(f"  buying_power    : {a.get('buying_power')}")
    print(f"  equity         : {a.get('equity')}")
    print(f"  pattern_day_trader: {a.get('pattern_day_trader')}")

    # market clock
    c = requests.get(f"{base}/clock", headers=headers, timeout=15).json()
    print(f"  market is_open : {c.get('is_open')}  next_open={c.get('next_open')}")

    # open positions
    p = requests.get(f"{base}/positions", headers=headers, timeout=15).json()
    print(f"  open positions : {len(p)}")


if __name__ == "__main__":
    main()
