# Project Analysis TODO

Date: 2026-03-06
Project: `asset_price_analysis`

## Critical

- [ ] Fix custom `dte_long` handling in monthly/DTE reporting.
  - Problem: plot source data is only stored when `dte == 23`, but report generation still expects it for monthly section.
  - Impact: non-23 user input can break report generation and produce wrong labels.
  - Refs:
    - `helper/data_analysis.py:169`
    - `helper/data_analysis.py:636`
    - `helper/data_analysis.py:643`
    - `helper/data_analysis.py:944`
    - `helper/data_analysis.py:1024`
    - 