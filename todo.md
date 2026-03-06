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

## High

- [ ] Guard cumulative probability function against empty input.
  - Problem: code accesses `data[0]` without checking length.
  - Impact: crashes when dataset is all-positive or all-negative after filtering.
  - Ref: `helper/data_analysis.py:404`

- [ ] Fix VIX fill logic out-of-bounds access.
  - Problem: fill loop reads `idx + 1` even at last element.
  - Impact: potential `IndexError` when latest row has missing VIX value.
  - Refs:
    - `helper/data_analysis.py:518`
    - `helper/data_analysis.py:520`

- [ ] Guard weekly summary min/max/mean calculations against empty lists.
  - Problem: `np.min/np.max/np.mean` used without empty checks on positive/negative buckets.
  - Impact: crashes in one-sided periods.
  - Refs:
    - `helper/data_analysis.py:1102`
    - `helper/data_analysis.py:1117`

## Medium

- [ ] Fix gap-down “beyond max gap” sign condition.
  - Problem: uses `< +2.5` instead of `< -2.5`.
  - Impact: wrong bucket assignment for negative-gap analysis.
  - Ref: `helper/data_analysis.py:353`

- [ ] Correct DTE drawdown formula denominator.
  - Problem: denominator is `lowest_low` instead of opening/reference price.
  - Impact: drawdown magnitude is distorted.
  - Ref: `helper/data_analysis.py:1324`

- [ ] Remove duplicated gap-up computation block (indentation/structure issue).
  - Problem: same stats are recomputed and overwritten inside inner loop.
  - Impact: unnecessary runtime and confusing logic.
  - Refs:
    - `helper/data_analysis.py:300`
    - `helper/data_analysis.py:303`
    - `helper/data_analysis.py:337`

## Low

- [ ] Expand test coverage beyond private helper methods.
  - Current tests validate only cumulative probability helpers.
  - Add integration/edge-case tests for date handling, weekly grouping, VIX filling, and report generation paths.
  - Ref: `tests/asset_analysis_utests.py`

- [ ] Improve repository hygiene.
  - Add root `.gitignore`.
  - Stop tracking generated artifacts: `venv/`, `__pycache__/`, local HTML outputs, `.DS_Store`.

## Project Snapshot

- Streamlit app entrypoint: `analyze_price_movement.py`
- Core analytics logic concentrated in one large class: `helper/data_analysis.py`
- Informational pages:
  - `pages/about_this_app.py`
  - `pages/contact.py`

## Suggested Execution Order

1. Critical correctness fixes (year/week grouping + custom DTE flow)
2. High-priority crash guards (empty lists, VIX fill bounds, weekly min/max checks)
3. Medium logic corrections (gap sign, drawdown formula, duplicate gap block)
4. Test additions for each bugfix
5. Repo cleanup (`.gitignore`, remove tracked generated files)
