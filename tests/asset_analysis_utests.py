# Copyright (c) 2024 Jacopo Ventura

import unittest
import datetime
import os
import re
import tempfile
from unittest.mock import patch
import pandas as pd
from helper.data_analysis import PriceAnalysis

dataset = {
    "positive": {
        "data": [1, 2, 3, 4, 5, 4, 1, 1, 0, 9],
        "bins":  [3, 6, 9],
        "cpf": [60., 90., 100.]
                },
    "negative": {
        "data": [-1, -2, -3, -4, -5, -4, -1, -1, 0, -9],
        "bins":  [-3, -6, -9],
        "cpf": [60., 90., 100.]
                },
    "positive_detailed": {
        "data": [1, 2, 3, 4, 5, 4, 1, 1, 0, 9],
        "bins": [1, 2, 3, 4],
        "cpf": [40., 50., 60., 80.]
    },
    "negative_detailed": {
        "data": [-1, -2, -3, -4, -5, -4, -1, -1, 0, -9],
        "bins": [-1, -2, -3, -4],
        "cpf": [40., 50., 60., 80.]
    },
}

start = datetime.datetime(2022, 1, 1)  # Year, Month, Day
end = datetime.datetime(2023, 2, 10)  # Year, Month, Day
dte = 23
spy = PriceAnalysis("SPY", start, end, dte, "/tmp/")


def build_asset_history_df() -> pd.DataFrame:
    """
    Build deterministic price history with both positive and negative daily moves.
    Index is intentionally ascending to validate sorting in query_asset_price.
    """
    dates = pd.bdate_range("2024-01-01", periods=15)
    close_list = [100, 101, 99, 102, 101, 103, 100, 104, 103, 105, 102, 106, 104, 107, 105]
    open_list = [100.2, 100.6, 99.4, 101.5, 101.2, 102.5, 100.5, 103.8, 102.6, 105.4, 101.7, 106.2, 103.5, 107.3, 104.8]
    low_list = [min(day_open, day_close) - 1.0 for day_open, day_close in zip(open_list, close_list)]
    high_list = [max(day_open, day_close) + 1.0 for day_open, day_close in zip(open_list, close_list)]
    return pd.DataFrame(
        {
            "Open": open_list,
            "High": high_list,
            "Low": low_list,
            "Close": close_list,
            "Volume": [1_000_000] * len(dates),
        },
        index=dates,
    )


class FakeTicker:
    def __init__(self, symbol: str, asset_df: pd.DataFrame, vix_df: pd.DataFrame | None):
        self._symbol = symbol
        self._asset_df = asset_df
        self._vix_df = vix_df

    def history(self, start=None, end=None) -> pd.DataFrame:
        if self._symbol == "^VIX":
            if self._vix_df is None:
                return pd.DataFrame()
            return self._vix_df.copy()
        return self._asset_df.copy()


def make_ticker_factory(asset_df: pd.DataFrame, vix_df: pd.DataFrame | None = None):
    def factory(symbol: str):
        return FakeTicker(symbol, asset_df, vix_df)

    return factory


# helper
def get_solution_from_dict(input_cpf: dict, input_bins: list) -> list:
    solution = []

    for bin_value in input_bins:
        for key in input_cpf.keys():
            n = re.findall("[\d\.\d]+", key)
            if n:
                if bin_value == float(n[0]):
                    solution.append(input_cpf[key])

    return solution


class TestCumulativeProbability(unittest.TestCase):

    def test_cpf_positive_data(self):
        cpf = spy._PriceAnalysis__calc_cpf(dataset["positive"]["data"], dataset["positive"]["bins"])
        self.assertEqual(dataset["positive"]["cpf"], cpf)

    def test_cpf_negative_data(self):
        cpf = spy._PriceAnalysis__calc_cpf([-i for i in dataset["negative"]["data"]], [-i for i in dataset["negative"]["bins"]])
        self.assertEqual(dataset["negative"]["cpf"], cpf)

    def test_cumulative_positive_data(self):
        cpf = spy._PriceAnalysis__calc_cumulative_probability(dataset["positive_detailed"]["data"])
        solution = get_solution_from_dict(cpf, dataset["positive_detailed"]["bins"])
        self.assertEqual(dataset["positive_detailed"]["cpf"], solution)

    def test_cumulative_negative_data(self):
        cpf = spy._PriceAnalysis__calc_cumulative_probability(dataset["negative_detailed"]["data"])
        solution = get_solution_from_dict(cpf, [-i for i in dataset["negative_detailed"]["bins"]])
        self.assertEqual(dataset["negative_detailed"]["cpf"], solution)

    def test_cumulative_empty_data(self):
        cpf = spy._PriceAnalysis__calc_cumulative_probability([])
        self.assertEqual(0, cpf["frequency [%]"])
        for pct in spy._PriceAnalysis__BINS_DAILY_CHANGE:
            key = str(int(pct * 10) / 10) + "% change"
            self.assertIn(key, cpf)
            self.assertEqual(0.0, cpf[key])

    def test_gapdown_beyond_max_gap_bucket_uses_negative_threshold(self):
        analysis = PriceAnalysis("SPY", start, end, dte, "/tmp/")
        analysis._PriceAnalysis__price_history_df = pd.DataFrame(
            {
                # No value is below -2.5, but several are below +2.5.
                "Open wrt close": [-1.0, -0.5, 0.4, 2.0],
                "Close wrt close": [-1.2, -0.2, 0.3, -0.4],
            }
        )

        analysis._PriceAnalysis__calc_stats_gapup_down()

        beyond_key = f">+{-analysis._PriceAnalysis__MAX_GAP} %"
        beyond_bucket = analysis._PriceAnalysis__stats_negative_gap[beyond_key]
        no_data = analysis._PriceAnalysis__NO__DATA_INDICATOR

        value_keys = [key for key in beyond_bucket.keys() if key != "gap"]
        self.assertGreater(len(value_keys), 0)
        for key in value_keys:
            self.assertEqual(
                no_data,
                beyond_bucket[key],
                msg=f"Expected no data for {key}, got {beyond_bucket[key]!r}",
            )


class TestPriceAnalysisIntegration(unittest.TestCase):
    def setUp(self):
        self.start = datetime.datetime(2024, 1, 1)
        self.end = datetime.datetime(2024, 2, 10)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.tmp_dir.name + "/"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_query_asset_price_normalizes_dates_and_week_fields(self):
        asset_df = build_asset_history_df()
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=False)

        with patch("helper.data_analysis.yf.Ticker", side_effect=make_ticker_factory(asset_df)):
            analysis.query_asset_price()

        price_df = analysis.get_price_history()
        self.assertEqual("yahoo", analysis.get_source())
        self.assertEqual(len(asset_df), len(price_df))
        self.assertGreater(price_df["Date"].iloc[0], price_df["Date"].iloc[-1])
        for key in ["Weekday", "Week number", "Year", "Open wrt close", "Close wrt close"]:
            self.assertIn(key, price_df.columns)

    def test_query_vix_fills_missing_values_without_out_of_bounds(self):
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=True)
        analysis._PriceAnalysis__price_history_df = pd.DataFrame(
            {
                "Date": [
                    datetime.date(2024, 1, 5),
                    datetime.date(2024, 1, 4),
                    datetime.date(2024, 1, 3),
                    datetime.date(2024, 1, 2),
                ],
                "Open": [100.0, 100.0, 100.0, 100.0],
                "Close": [101.0, 101.0, 101.0, 101.0],
            }
        )
        vix_df = pd.DataFrame(
            {"Close": [20.0]},
            index=pd.to_datetime(["2024-01-04"]),
        )

        with patch("helper.data_analysis.yf.Ticker", side_effect=make_ticker_factory(pd.DataFrame(), vix_df)):
            analysis.query_vix()

        self.assertEqual([20.0, 20.0, 0.0, 0.0], analysis.get_price_history()["VIX"].tolist())

    def test_weekly_grouping_handles_one_sided_data(self):
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=False)
        dates_desc = [day.date() for day in pd.bdate_range("2024-01-08", "2024-01-19")[::-1]]
        week_numbers = [pd.Timestamp(day).isocalendar().week for day in dates_desc]
        years = [day.year for day in dates_desc]
        analysis._PriceAnalysis__price_history_df = pd.DataFrame(
            {
                "Date": dates_desc,
                "Week number": week_numbers,
                "Year": years,
                "Open": [100.0] * len(dates_desc),
                "Close": [101.0] * len(dates_desc),
                "Low": [99.0] * len(dates_desc),
                "VIX": [12.0] * len(dates_desc),
            }
        )
        analysis._PriceAnalysis__years_list = sorted(set(years))

        analysis._PriceAnalysis__calc_weekly_statistics()
        weekly_df = analysis._PriceAnalysis__weekly_change_df

        for row_name in ["Monday to Friday: negative", "Friday to Friday: negative"]:
            self.assertEqual(0.0, float(weekly_df.loc[row_name, "Max drawdown [%]"]))
            self.assertEqual(0.0, float(weekly_df.loc[row_name, "Avg drawdown [%]"]))
            self.assertEqual(0.0, float(weekly_df.loc[row_name, "Max VIX increment [%]"]))
            self.assertEqual(0.0, float(weekly_df.loc[row_name, "Avg VIX increment [%]"]))

    def test_run_generates_html_report_with_mocked_data(self):
        asset_df = build_asset_history_df()
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=False, do_plot=False)

        with patch("helper.data_analysis.yf.Ticker", side_effect=make_ticker_factory(asset_df)):
            analysis.run()

        self.assertTrue(os.path.exists(analysis.FILENAME))
        with open(analysis.FILENAME, "r") as output_file:
            html = output_file.read()

        self.assertIn("Daily and weekly change stats", html)
        self.assertIn("Open gap-up / down analysis", html)
        self.assertIn("Daily change according to VIX", html)
        self.assertTrue((analysis.get_price_history()["VIX"] == 0.0).all())

    def test_weekly_conditional_stats_group_by_year_and_week(self):
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=False)

        week_2024_dates = pd.bdate_range("2024-01-08", periods=5)  # ISO week 2
        week_2025_dates = pd.bdate_range("2025-01-06", periods=5)  # ISO week 2

        week_2024 = pd.DataFrame(
            {
                "Date": [d.date() for d in week_2024_dates],
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "Close": [101.0, 102.0, 103.0, 104.0, 105.0],  # positive week
                "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "VIX": [15.0] * 5,
            }
        )
        week_2025 = pd.DataFrame(
            {
                "Date": [d.date() for d in week_2025_dates],
                "Open": [100.0, 101.0, 100.0, 99.0, 97.0],
                "Close": [101.0, 100.0, 99.0, 98.0, 96.0],  # negative week, but Monday positive
                "Low": [99.0, 99.0, 98.0, 97.0, 95.0],
                "VIX": [18.0] * 5,
            }
        )

        weekly_df = pd.concat([week_2024, week_2025], ignore_index=True)
        weekly_df["Week number"] = [pd.Timestamp(day).isocalendar().week for day in weekly_df["Date"]]
        weekly_df["Year"] = [day.year for day in weekly_df["Date"]]
        # Mimic production ordering (most recent first)
        weekly_df = weekly_df.sort_values("Date", ascending=False).reset_index(drop=True)

        analysis._PriceAnalysis__price_history_df = weekly_df
        analysis._PriceAnalysis__years_list = sorted(set(weekly_df["Year"]))
        analysis._PriceAnalysis__calc_weekly_conditional_statistics()

        conditional_df = analysis._PriceAnalysis__weekly_change_monday_conditional_df
        self.assertEqual(50.0, float(conditional_df.loc["Week if Monday positive: positive", "frequency [%]"]))
        self.assertEqual(50.0, float(conditional_df.loc["Week if Monday positive: negative", "frequency [%]"]))


if __name__ == '__main__':
    unittest.main()
