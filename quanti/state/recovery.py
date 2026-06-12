"""
Crash recovery logic.
On startup, reads the last checkpoint and replays the journal to restore state.
"""

from datetime import datetime

from quanti.state.journal import Journal


def recover_portfolio(journal: Journal) -> dict:
    """
    Recover portfolio state after a crash.

    Strategy:
    1. Read last checkpoint (full snapshot)
    2. Replay all journal entries since checkpoint timestamp
    3. Return recovered portfolio state

    Returns:
        {
            "positions": {symbol: {"quantity": int, "avg_cost": float}},
            "cash": float | None,   # None = no checkpoint, can't determine cash
            "checkpoint_ts": str,
            "replayed_entries": int,
        }
    """
    cp = journal.get_last_checkpoint()

    if cp is None:
        print("WARN: No checkpoint found. Computing positions from full journal.")
        computed = journal.compute_current_positions()
        return {
            "positions": computed,
            "cash": None,   # Sentinel: no checkpoint means cash is unknowable
            "checkpoint_ts": None,
            "replayed_entries": 0,
        }

    checkpoint_ts, snapshot = cp
    cash = snapshot.get("cash", 0.0)
    positions_snapshot = snapshot.get("positions", {})

    # Normalize snapshot positions to uniform internal format:
    #   {"quantity": int, "total_cost": float, "avg_cost": float}
    # Checkpoints may store {"quantity": q, "avg_cost": ac} without total_cost.
    positions: dict[str, dict] = {}
    for sym, pos in positions_snapshot.items():
        qty = pos.get("quantity", 0) if isinstance(pos, dict) else 0
        if isinstance(qty, dict):
            qty = qty.get("quantity", 0)
        avg_cost = pos.get("avg_cost", 0.0) if isinstance(pos, dict) else 0.0
        if isinstance(avg_cost, dict):
            avg_cost = avg_cost.get("avg_cost", 0.0)
        positions[sym] = {
            "quantity": qty,
            "total_cost": qty * avg_cost,
            "avg_cost": avg_cost,
        }

    # Replay entries since checkpoint
    entries = journal.get_journal_since(checkpoint_ts)
    for entry in entries:
        sym = entry["symbol"]
        side = entry["side"]
        qty = entry["quantity"]
        price = entry["price"]

        if sym not in positions:
            positions[sym] = {"quantity": 0, "total_cost": 0.0, "avg_cost": 0.0}

        if side == "buy":
            positions[sym]["quantity"] += qty
            positions[sym]["total_cost"] += qty * price
            if positions[sym]["quantity"] > 0:
                positions[sym]["avg_cost"] = positions[sym]["total_cost"] / positions[sym]["quantity"]
            cash -= qty * price
        elif side == "sell":
            # Reduce cost proportionally before removing quantity
            old_qty = positions[sym]["quantity"]
            if old_qty > 0 and qty > 0:
                ratio = qty / old_qty
                positions[sym]["total_cost"] *= (1 - ratio)
            positions[sym]["quantity"] -= qty
            if positions[sym]["quantity"] > 0:
                positions[sym]["avg_cost"] = positions[sym]["total_cost"] / positions[sym]["quantity"]
            cash += qty * price

    # Format output with just quantity + avg_cost
    result_positions = {}
    for sym, pos in positions.items():
        qty = pos.get("quantity", 0)
        if qty > 0:
            result_positions[sym] = {
                "quantity": qty,
                "avg_cost": pos.get("avg_cost", 0.0),
            }

    return {
        "positions": result_positions,
        "cash": cash,
        "checkpoint_ts": checkpoint_ts,
        "replayed_entries": len(entries),
    }


def build_checkpoint_snapshot(
    positions: dict,
    cash: float,
    pending_orders: list[dict],
) -> dict:
    """
    Build a snapshot dict for checkpointing.

    Args:
        positions: {symbol: {"quantity": int, "avg_cost": float}}
        cash: Available cash
        pending_orders: List of non-terminal order dicts from journal

    Returns:
        JSON-serializable snapshot dict
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "cash": cash,
        "positions": positions,
        "pending_order_ids": [o["order_id"] for o in pending_orders],
    }
