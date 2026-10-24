"""CSV logging for simulated experiments only; never labels simulation as field data."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import MODEL_SCHEMA_VERSION

FIELDS = [
    "timestamp_utc", "run_id", "step", "source_type", "model_schema_version",
    "frequency_hz", "df_dt_hz_s", "charge_current_a", "battery_temp_c",
    "house_load_kw", "target_charge_current_a", "mechanical_power_kw",
    "action_requested_a", "action_applied_delta_a", "next_frequency_hz",
    "next_df_dt_hz_s", "next_charge_current_a", "next_battery_temp_c",
    "next_house_load_kw", "next_target_charge_current_a", "next_mechanical_power_kw",
    "reward", "terminated", "truncated", "guard_active", "total_load_kw", "overload_kw",
]


def _as_float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def log_simulated_experience(
    path: str | Path,
    *,
    run_id: str,
    step: int,
    obs: np.ndarray,
    next_obs: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> None:
    """Append one explicitly simulated transition to a CSV file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not target.exists() or target.stat().st_size == 0
    values = [
        datetime.now(timezone.utc).isoformat(), run_id, step, "simulation",
        MODEL_SCHEMA_VERSION,
        _as_float(obs[0]), _as_float(obs[1]), _as_float(obs[2]), _as_float(obs[3]),
        _as_float(obs[4]), _as_float(obs[5]), _as_float(obs[6]),
        info.get("action_requested_a", 0.0), info.get("action_applied_delta_a", 0.0),
        _as_float(next_obs[0]), _as_float(next_obs[1]), _as_float(next_obs[2]),
        _as_float(next_obs[3]), _as_float(next_obs[4]), _as_float(next_obs[5]),
        _as_float(next_obs[6]), reward, terminated, truncated, info.get("guard_active", False),
        info.get("total_load_kw", ""), info.get("overload_kw", ""),
    ]
    with target.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if needs_header:
            writer.writerow(FIELDS)
        writer.writerow(values)
