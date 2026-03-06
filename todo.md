# Project Analysis TODO

Date: 2026-03-06
Project: `asset_price_analysis`

## Critical

- [ ] Fix weekly calculations mixing data across years.
  - Problem: filters use `Week number` only, not `(Year, Week number)`, so week 1 of different years is merged.
  - Impact: weekly stats and conditional weekly logic are incorrect for multi-year analysis windows.
  - Refs:
    - `helper/data_analysis.py:911`
    - `helper/data_analysis.py:1173`
    - `helper/data_analysis.py:1232`
    - `helper/data_analysis.py:1265`
    - `helper/data_analysis.py:1267`

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