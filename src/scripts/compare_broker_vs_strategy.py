#!/usr/bin/env python3
"""
End-to-end comparison tool: Broker orders -> paired trades, then match
against strategy trades with configurable time tolerances. Produces
matches, unmatched with reasons, and summary JSONs.

Usage example:
  python -m src.scripts.compare_broker_vs_strategy \
    --orders broker_trades.csv \
    --run-dir outputs/20250908_052243/open_source_baseline_reference/2025-05-29_to-2025-09-06 \
    --start 2025-08-26 --end 2025-09-04 \
    --entry-tol 20 --exit-tol 20 --exit-enforce true
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare broker vs strategy trades end-to-end")
    p.add_argument("--orders", required=True, help="Path to broker orders CSV (order-format)")
    p.add_argument("--run-dir", required=True, help="Backtest run directory (root of outputs for a run)")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    p.add_argument("--entry-tol", type=int, default=20, help="Entry tolerance in minutes (default 20)")
    p.add_argument("--exit-tol", type=int, default=20, help="Exit tolerance in minutes (default 20)")
    p.add_argument("--exit-enforce", type=str, default="true", help="Enforce exit tolerance: true/false")
    p.add_argument("--out-subdir", default="audits/comparison_e2e", help="Output subdir under run-dir")
    return p.parse_args()


def clean_symbol(sym: str) -> str:
    s = str(sym)
    for suf in ("-EQ", "-BE", "-BL", "-E1", "-E2"):
        s = s.replace(suf, "")
    return s


def load_broker_orders(orders_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(orders_csv)
    # Normalize common columns
    colmap = {
        "TransactionType": "side",
        "Trade Date": "time",
        "Price": "price",
        "Quantity": "qty",
        "Symbol": "symbol",
    }
    for k, v in colmap.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "symbol", "side"]).copy()
    df["ticker"] = df["symbol"].apply(clean_symbol)
    df["side"] = df["side"].astype(str).str.upper().str.strip()
    return df


def pair_broker_trades_cross_day(orders: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(f"{start} 00:00:00")
    end_ts = pd.Timestamp(f"{end} 23:59:59")
    df = orders[(orders["time"] >= start_ts) & (orders["time"] <= end_ts)].copy()
    rows: List[Dict[str, Any]] = []
    qty_mismatch = 0
    # Maintain rolling open per ticker across the window
    for tic, grp in df.sort_values("time").groupby("ticker"):
        open_order = None
        for _, r in grp.iterrows():
            side = r["side"]
            t = r["time"]
            px = float(r["price"]) if pd.notna(r["price"]) else None
            qty = int(r["qty"]) if "qty" in r and pd.notna(r["qty"]) else 1
            if open_order is None:
                open_order = dict(side=side, time=t, price=px, qty=qty, symbol=r["symbol"])
                continue
            if side == open_order["side"]:
                # same side: keep earliest as entry
                continue
            q = min(open_order["qty"], qty)
            if q != open_order["qty"] or q != qty:
                qty_mismatch += 1
            entry_side = open_order["side"].title()
            rows.append(
                {
                    "source": "broker",
                    "ticker": tic,
                    "symbol": open_order["symbol"],
                    "trade_type": entry_side,
                    "entry_time": open_order["time"],
                    "entry_price": open_order["price"],
                    "exit_time": t,
                    "exit_price": px,
                    "quantity": q,
                }
            )
            open_order = None
    out = pd.DataFrame(rows)
    return out


def load_strategy_window(run_dir: Path, start: str, end: str) -> pd.DataFrame:
    merged_win = run_dir / "audits/merged/STRATEGY_AllTrades_{}{}_to_{}{}.csv".format(start, "", end, "")
    # The file is already produced by previous steps; otherwise fall back to assemble from per-ticker
    if not merged_win.exists():
        # Build from per-ticker strategy trades
        strat_dir = run_dir / "data/strategy_trades"
        parts = []
        for p in strat_dir.glob("*_StrategyTrades_*.csv"):
            df = pd.read_csv(p)
            for c in ["Entry Time", "Exit Time"]:
                s = pd.to_datetime(df[c], errors="coerce")
                if s.dt.tz is not None:
                    s = s.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
                df[c] = s
            parts.append(df)
        all_trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        start_ts = pd.Timestamp(f"{start} 00:00:00")
        end_ts = pd.Timestamp(f"{end} 23:59:59")
        window = all_trades[(all_trades["Entry Time"] >= start_ts) & (all_trades["Entry Time"] <= end_ts)].copy()
        return window
    else:
        df = pd.read_csv(merged_win, parse_dates=["Entry Time", "Exit Time"])
        return df


def match_trades(
    broker_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
    entry_tol_min: int,
    exit_tol_min: int,
    exit_enforce: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    entry_tol = pd.Timedelta(minutes=entry_tol_min)
    exit_tol = pd.Timedelta(minutes=exit_tol_min)

    # Map strategy to unified schema
    s_unified = pd.DataFrame(
        {
            "source": "strategy",
            "ticker": strategy_df["ticker"],
            "symbol": strategy_df["ticker"].astype(str) + "-EQ",
            "trade_type": strategy_df["Trade Type"].astype(str).str.title(),
            "entry_time": strategy_df["Entry Time"],
            "entry_price": strategy_df["Entry Price"],
            "exit_time": strategy_df["Exit Time"],
            "exit_price": strategy_df["Exit Price"],
            "quantity": 1,
        }
    )

    # Index by (ticker, side)
    by_key: Dict[Tuple[str, str], List[Tuple[int, pd.Series]]] = {}
    for idx, row in s_unified.iterrows():
        by_key.setdefault((row["ticker"], row["trade_type"]), []).append((idx, row))

    matches: List[Dict[str, Any]] = []
    unmatched_broker: List[Dict[str, Any]] = []
    unmatched_strategy = set(range(len(s_unified)))

    for _, b in broker_df.iterrows():
        key = (b["ticker"], b["trade_type"])
        cands = by_key.get(key, [])
        best = None
        best_score = None
        best_exit_delta = None
        for s_idx, s in cands:
            if pd.isna(s["entry_time"]) or pd.isna(b["entry_time"]):
                continue
            dt_entry = abs(s["entry_time"] - b["entry_time"])
            if dt_entry > entry_tol:
                continue
            if pd.notna(s["exit_time"]) and pd.notna(b["exit_time"]):
                dt_exit = abs(s["exit_time"] - b["exit_time"])
            else:
                dt_exit = pd.NaT
            score = dt_entry + (dt_exit if isinstance(dt_exit, pd.Timedelta) else pd.Timedelta(0))
            if best is None or score < best_score:
                best = (s_idx, s, dt_entry, dt_exit)
                best_score = score
                best_exit_delta = dt_exit
        if best is None:
            unmatched_broker.append({
                "ticker": b["ticker"],
                "trade_type": b["trade_type"],
                "entry_time": b["entry_time"],
                "exit_time": b["exit_time"],
                "reason": "no_strategy_candidate_within_entry_tolerance"
            })
        else:
            s_idx, s, dt_entry, dt_exit = best
            if exit_enforce and isinstance(dt_exit, pd.Timedelta) and dt_exit > exit_tol:
                unmatched_broker.append({
                    "ticker": b["ticker"],
                    "trade_type": b["trade_type"],
                    "entry_time": b["entry_time"],
                    "exit_time": b["exit_time"],
                    "reason": f"exit_delta_exceeds_{exit_tol_min}m",
                    "entry_delta_min": round(dt_entry.total_seconds() / 60.0, 2),
                    "exit_delta_min": round(dt_exit.total_seconds() / 60.0, 2),
                })
            else:
                if s_idx in unmatched_strategy:
                    unmatched_strategy.remove(s_idx)
                matches.append({
                    "ticker": b["ticker"],
                    "trade_type": b["trade_type"],
                    "broker_entry": b["entry_time"],
                    "strategy_entry": s["entry_time"],
                    "entry_delta_min": round(dt_entry.total_seconds() / 60.0, 2),
                    "broker_exit": b["exit_time"],
                    "strategy_exit": s["exit_time"],
                    "exit_delta_min": (round(dt_exit.total_seconds() / 60.0, 2) if isinstance(dt_exit, pd.Timedelta) else None),
                    "broker_entry_price": b["entry_price"],
                    "strategy_entry_price": s["entry_price"],
                    "broker_exit_price": b["exit_price"],
                    "strategy_exit_price": s["exit_price"],
                })

    match_df = pd.DataFrame(matches)
    unmatched_broker_df = pd.DataFrame(unmatched_broker)
    unmatched_strategy_df = s_unified.iloc[list(unmatched_strategy)].copy()

    summary = {
        "entry_tolerance_min": entry_tol_min,
        "exit_tolerance_min": exit_tol_min,
        "exit_enforce": exit_enforce,
        "counts": {
            "broker_pairs": int(len(broker_df)),
            "strategy_trades_in_window": int(len(s_unified)),
            "matched": int(len(match_df)),
            "unmatched_broker": int(len(unmatched_broker_df)),
            "unmatched_strategy": int(len(unmatched_strategy_df)),
        },
    }
    return match_df, unmatched_broker_df, unmatched_strategy_df, summary


def to_json(p: Path, obj: Any) -> None:
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    orders = load_broker_orders(Path(args.orders))
    broker_trades = pair_broker_trades_cross_day(orders, args.start, args.end)
    broker_csv = out_dir / f"BROKER_Paired_Trades_{args.start}_to_{args.end}.csv"
    broker_trades.to_csv(broker_csv, index=False)

    strategy_win = load_strategy_window(run_dir, args.start, args.end)

    # 1) Entry-only enforcement (exit not enforced)
    m1, ub1, us1, s1 = match_trades(
        broker_trades, strategy_win, args.entry_tol, args.exit_tol, exit_enforce=False
    )
    m1.to_csv(out_dir / f"MATCHES_entry_only_{args.start}_to_{args.end}_{args.entry_tol}m.csv", index=False)
    ub1.to_csv(out_dir / f"UNMATCHED_broker_entry_only_{args.start}_to_{args.end}_{args.entry_tol}m.csv", index=False)
    us1.to_csv(out_dir / f"UNMATCHED_strategy_entry_only_{args.start}_to_{args.end}_{args.entry_tol}m.csv", index=False)
    to_json(out_dir / f"SUMMARY_entry_only_{args.start}_to_{args.end}_{args.entry_tol}m.json", s1)

    # 2) Entry+Exit enforcement
    m2, ub2, us2, s2 = match_trades(
        broker_trades, strategy_win, args.entry_tol, args.exit_tol, exit_enforce=(args.exit_enforce.lower()=="true")
    )
    m2.to_csv(out_dir / f"MATCHES_entry_exit_{args.start}_to_{args.end}_{args.entry_tol}m_{args.exit_tol}m.csv", index=False)
    ub2.to_csv(out_dir / f"UNMATCHED_broker_entry_exit_{args.start}_to_{args.end}_{args.entry_tol}m_{args.exit_tol}m.csv", index=False)
    us2.to_csv(out_dir / f"UNMATCHED_strategy_entry_exit_{args.start}_to_{args.end}_{args.entry_tol}m_{args.exit_tol}m.csv", index=False)
    to_json(out_dir / f"SUMMARY_entry_exit_{args.start}_to_{args.end}_{args.entry_tol}m_{args.exit_tol}m.json", s2)

    # 3) Per-ticker side counts for quick sanity
    def counts(df: pd.DataFrame, side_col: str) -> Dict[str, Any]:
        if df.empty:
            return {"total": 0, "by_ticker": {}, "by_side": {}}
        return {
            "total": int(len(df)),
            "by_ticker": {k: int(v) for k, v in df.groupby("ticker").size().to_dict().items()},
            "by_side": {k: int(v) for k, v in df.groupby(side_col).size().to_dict().items()},
        }

    overview = {
        "broker_pairs": counts(broker_trades, "trade_type"),
        "strategy_trades_in_window": counts(
            pd.DataFrame(
                {
                    "ticker": strategy_win.get("ticker", pd.Series(dtype=str)),
                    "Trade Type": strategy_win.get("Trade Type", pd.Series(dtype=str)),
                }
            ),
            "Trade Type",
        ),
        "matched_entry_only": counts(m1.rename(columns={"trade_type": "Trade Type"}), "Trade Type"),
        "matched_entry_exit": counts(m2.rename(columns={"trade_type": "Trade Type"}), "Trade Type"),
    }
    to_json(out_dir / f"OVERVIEW_{args.start}_to_{args.end}.json", overview)

    print("Comparison complete. Outputs written to:", out_dir)


if __name__ == "__main__":
    main()

