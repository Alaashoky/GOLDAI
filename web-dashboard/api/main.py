"""GOLDAI Web Dashboard API.

FastAPI backend serving bot data from local files.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GOLDAI Dashboard API",
    description="Serves GOLDAI trading bot data for the web dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data Directory ───────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _load_json(filename: str) -> dict[str, Any] | None:
    path = DATA_DIR / filename
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _load_csv(filename: str) -> list[dict[str, str]]:
    path = DATA_DIR / filename
    if path.exists():
        try:
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except OSError:
            return []
    return []


# ─── Mock Data Helpers ────────────────────────────────────────────────────────

def _mock_status() -> dict[str, Any]:
    return {
        "is_running": False,
        "last_run": datetime.utcnow().isoformat(),
        "scan_interval": 300,
        "current_regime": "RANGING",
        "session": "LONDON",
        "account_balance": 10000.00,
        "daily_pnl": 0.00,
        "win_rate": 0.0,
        "open_positions_count": 0,
        "_mock": True,
    }


def _mock_positions() -> list[dict[str, Any]]:
    return []


def _mock_trades() -> list[dict[str, Any]]:
    base_time = datetime.utcnow()
    mock = []
    directions = ["BUY", "SELL"]
    profits = [125.50, -45.20, 88.30, 200.10, -30.00, 55.75, 140.00, -22.50, 95.40, 178.60]
    for i, profit in enumerate(profits):
        open_time = base_time - timedelta(hours=i * 3 + 1)
        close_time = open_time + timedelta(hours=1, minutes=30)
        direction = directions[i % 2]
        entry = 2050.00 + (i * 5)
        exit_price = entry + (15 if profit > 0 else -10)
        mock.append({
            "ticket": 1000 + i,
            "symbol": "XAUUSD",
            "direction": direction,
            "entry_price": round(entry, 2),
            "exit_price": round(exit_price, 2),
            "profit": profit,
            "volume": 0.01,
            "open_time": open_time.isoformat(),
            "close_time": close_time.isoformat(),
            "duration": "1h 30m",
        })
    return mock


def _mock_equity_curve() -> list[dict[str, Any]]:
    base = 10000.0
    equity = base
    points = []
    now = datetime.utcnow()
    for i in range(30):
        dt = now - timedelta(days=29 - i)
        change = (hash(str(i)) % 200 - 100) * 0.5
        equity = max(equity + change, base * 0.8)
        points.append({
            "time": dt.strftime("%b %d"),
            "equity": round(equity, 2),
            "balance": round(equity - abs(change) * 0.3, 2),
        })
    return points


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/status")
async def get_status() -> dict[str, Any]:
    data = _load_json("status.json")
    if data:
        return data
    return _mock_status()


@app.get("/positions")
async def get_positions() -> dict[str, Any]:
    data = _load_json("positions.json")
    if data:
        return {"positions": data.get("positions", data if isinstance(data, list) else [])}

    # Attempt MT5 live data (optional — skipped if unavailable)
    try:
        import MetaTrader5 as mt5  # type: ignore

        if mt5.initialize():
            raw = mt5.positions_get()
            mt5.shutdown()
            if raw:
                positions = [
                    {
                        "ticket": int(p.ticket),
                        "symbol": p.symbol,
                        "direction": "BUY" if p.type == 0 else "SELL",
                        "entry_price": float(p.price_open),
                        "stop_loss": float(p.sl),
                        "take_profit": float(p.tp),
                        "current_profit": float(p.profit),
                        "volume": float(p.volume),
                        "open_time": datetime.fromtimestamp(p.time).isoformat(),
                    }
                    for p in raw
                ]
                return {"positions": positions}
    except ImportError:
        pass

    return {"positions": _mock_positions()}


@app.get("/trades")
async def get_trades() -> dict[str, Any]:
    rows = _load_csv("trades.csv")
    if rows:
        trades: list[dict[str, Any]] = []
        for row in rows[-10:]:
            try:
                open_time = row.get("open_time", "")
                close_time = row.get("close_time", "")
                duration = "—"
                if open_time and close_time:
                    try:
                        ot = datetime.fromisoformat(open_time)
                        ct = datetime.fromisoformat(close_time)
                        diff = ct - ot
                        h, rem = divmod(int(diff.total_seconds()), 3600)
                        m = rem // 60
                        duration = f"{h}h {m}m" if h else f"{m}m"
                    except ValueError:
                        pass

                trades.append({
                    "ticket": int(row.get("ticket", 0)),
                    "symbol": row.get("symbol", "XAUUSD"),
                    "direction": row.get("direction", "BUY"),
                    "entry_price": float(row.get("entry_price", 0)),
                    "exit_price": float(row.get("exit_price", 0)),
                    "profit": float(row.get("profit", 0)),
                    "volume": float(row.get("volume", 0.01)),
                    "open_time": open_time,
                    "close_time": close_time,
                    "duration": duration,
                })
            except (KeyError, ValueError):
                continue
        return {"trades": trades}

    # Fallback: check JSON
    data = _load_json("trades.json")
    if data:
        trades_list = data.get("trades", data if isinstance(data, list) else [])
        return {"trades": trades_list}

    return {"trades": _mock_trades()}


@app.get("/equity-curve")
async def get_equity_curve() -> dict[str, Any]:
    data = _load_json("equity_curve.json")
    if data:
        return {"equity_curve": data.get("equity_curve", data if isinstance(data, list) else [])}

    rows = _load_csv("equity_curve.csv")
    if rows:
        points = []
        for row in rows:
            try:
                points.append({
                    "time": row.get("time", ""),
                    "equity": float(row.get("equity", 0)),
                    "balance": float(row.get("balance", 0)),
                })
            except (KeyError, ValueError):
                continue
        if points:
            return {"equity_curve": points}

    return {"equity_curve": _mock_equity_curve()}


@app.get("/performance")
async def get_performance() -> dict[str, Any]:
    data = _load_json("performance.json")
    if data:
        return data

    trades_data = await get_trades()
    trades = trades_data.get("trades", [])
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    wins = [t for t in trades if float(t.get("profit", 0)) > 0]
    losses = [t for t in trades if float(t.get("profit", 0)) <= 0]
    total_profit = sum(float(t.get("profit", 0)) for t in trades)
    gross_profit = sum(float(t.get("profit", 0)) for t in wins)
    gross_loss = abs(sum(float(t.get("profit", 0)) for t in losses))

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "total_profit": round(total_profit, 2),
        "avg_profit": round(gross_profit / len(wins), 2) if wins else 0.0,
        "avg_loss": round(gross_loss / len(losses), 2) if losses else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
        "max_drawdown": 0.0,
    }


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
