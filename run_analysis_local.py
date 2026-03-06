#!/usr/bin/env python3

import datetime as dt
from pathlib import Path
import sys
import time

from helper.data_analysis import PriceAnalysis
from yfinance.exceptions import YFRateLimitError

# Edit these values directly.
TICKER = "SPY"
START_DATE = dt.datetime(2020, 1, 1)
END_DATE = dt.datetime(2026, 2, 26)
DTE_LONG = 23
OUTPUT_DIR = "."
DO_PLOT = True
ENABLE_VIX_QUERY = True
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 10


def main() -> int:
    if END_DATE <= START_DATE:
        raise ValueError("END_DATE must be later than START_DATE.")
    if DTE_LONG <= 0:
        raise ValueError("DTE_LONG must be > 0.")

    output_dir = Path(OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path_to_report = str(output_dir) + "/"

    analysis = None
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            analysis = PriceAnalysis(
                ticker=TICKER.upper(),
                start=START_DATE,
                end=END_DATE,
                dte_long=DTE_LONG,
                path_to_report=path_to_report,
                stats_vix=ENABLE_VIX_QUERY,
                do_plot=DO_PLOT,
            )
            analysis.run()
            break
        except YFRateLimitError as error:
            last_error = error
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Yahoo Finance rate limit persisted after {MAX_RETRIES} attempts. "
                    "Wait a few minutes and run again."
                ) from error
            wait_seconds = RETRY_BASE_SECONDS * attempt
            print(
                f"Rate limited by Yahoo Finance (attempt {attempt}/{MAX_RETRIES}). "
                f"Retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)

    if analysis is None:
        raise RuntimeError("Analysis failed before initialization.") from last_error

    report_path = Path(analysis.FILENAME).expanduser().resolve()
    print(f"Report written to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
