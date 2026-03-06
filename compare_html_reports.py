#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import pandas as pd


# Edit these paths if needed.
REFERENCE_FILE = "SPY_01012020_to_26022026_reference.html"
CURRENT_FILE = "SPY_01012020_to_26022026_reference.html"

FLOAT_TOLERANCE = 0.1
MAX_DIFFS_PER_TABLE = 20


@dataclass
class TableDiff:
    table_index: int
    message: str


def to_float(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, int | float):
        return float(value)

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None

    # Keep this conservative: only treat plain numeric content as float.
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def compare_tables(reference_df: pd.DataFrame, current_df: pd.DataFrame, table_index: int) -> list[TableDiff]:
    diffs: list[TableDiff] = []

    ref_cols = [str(c).strip() for c in reference_df.columns]
    cur_cols = [str(c).strip() for c in current_df.columns]
    if ref_cols != cur_cols:
        diffs.append(
            TableDiff(
                table_index,
                f"columns differ\n  ref: {ref_cols}\n  cur: {cur_cols}",
            )
        )

    if reference_df.shape != current_df.shape:
        diffs.append(
            TableDiff(
                table_index,
                f"shape differs (ref={reference_df.shape}, cur={current_df.shape})",
            )
        )

    row_count = min(len(reference_df), len(current_df))
    col_count = min(len(reference_df.columns), len(current_df.columns))

    diff_count = 0
    for row_idx in range(row_count):
        for col_idx in range(col_count):
            ref_val = reference_df.iat[row_idx, col_idx]
            cur_val = current_df.iat[row_idx, col_idx]

            ref_float = to_float(ref_val)
            cur_float = to_float(cur_val)

            if ref_float is not None and cur_float is not None:
                if abs(ref_float - cur_float) < FLOAT_TOLERANCE:
                    continue
            else:
                if str(ref_val).strip() == str(cur_val).strip():
                    continue

            diffs.append(
                TableDiff(
                    table_index,
                    f"row {row_idx}, col {col_idx} ({reference_df.columns[col_idx]}): "
                    f"ref={ref_val!r} cur={cur_val!r}",
                )
            )
            diff_count += 1
            if diff_count >= MAX_DIFFS_PER_TABLE:
                diffs.append(
                    TableDiff(
                        table_index,
                        f"reached max printed diffs ({MAX_DIFFS_PER_TABLE}) for this table",
                    )
                )
                return diffs

    return diffs


def main() -> int:
    reference_path = Path(REFERENCE_FILE)
    current_path = Path(CURRENT_FILE)

    if not reference_path.exists():
        print(f"Missing reference file: {reference_path}")
        return 1
    if not current_path.exists():
        print(f"Missing current file: {current_path}")
        return 1

    reference_tables = pd.read_html(reference_path)
    current_tables = pd.read_html(current_path)

    print(f"Reference tables: {len(reference_tables)}")
    print(f"Current tables:   {len(current_tables)}")

    all_diffs: list[TableDiff] = []

    if len(reference_tables) != len(current_tables):
        all_diffs.append(
            TableDiff(
                -1,
                f"table count differs (ref={len(reference_tables)}, cur={len(current_tables)})",
            )
        )

    for idx, (reference_df, current_df) in enumerate(zip(reference_tables, current_tables), start=1):
        table_diffs = compare_tables(reference_df, current_df, idx)
        all_diffs.extend(table_diffs)

    if not all_diffs:
        print("OK: reports match on all parsed tables.")
        return 0

    print("\nDifferences found:")
    for diff in all_diffs:
        if diff.table_index == -1:
            print(f"- {diff.message}")
        else:
            print(f"- Table {diff.table_index}: {diff.message}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
