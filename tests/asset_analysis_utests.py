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


def build_monotonic_bull_history_df(periods: int = 40) -> pd.DataFrame:
    """
    Build descending-ordered dataframe (production-like ordering) where price
    strictly increases forward in time.
    """
    dates_asc = pd.bdate_range("2024-01-01", periods=periods)
    close_asc = [100.0 + idx for idx in range(periods)]
    open_asc = [close - 0.3 for close in close_asc]
    low_asc = [day_open - 0.5 for day_open in open_asc]

    dates_desc = [date.date() for date in dates_asc[::-1]]
    close_desc = close_asc[::-1]
    open_desc = open_asc[::-1]
    low_desc = low_asc[::-1]

    return pd.DataFrame(
        {
            "Date": dates_desc,
            "Week number": [pd.Timestamp(day).isocalendar().week for day in dates_desc],
            "Year": [pd.Timestamp(day).isocalendar().year for day in dates_desc],
            "Open": open_desc,
            "Close": close_desc,
            "Low": low_desc,
            "VIX": [20.0] * periods,
        }
    )


def build_new_year_crossing_week_df() -> pd.DataFrame:
    """
    Build one ISO week that crosses calendar years (Dec -> Jan).
    """
    dates = pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-01", "2025-01-02", "2025-01-03"])
    close_list = [100.0, 101.0, 102.0, 103.0, 104.0]
    open_list = [close - 0.2 for close in close_list]
    low_list = [day_open - 0.5 for day_open in open_list]
    high_list = [close + 0.5 for close in close_list]
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
        self.assertEqual(0, cpf["count days"])
        for pct in spy._PriceAnalysis__BINS_DAILY_CHANGE:
            key = str(int(pct * 10) / 10) + "% change"
            self.assertIn(key, cpf)
            self.assertEqual(0.0, cpf[key])

    def test_distribution_includes_count_days(self):
        positive, negative = spy._PriceAnalysis__calc_distribution([1.0, -1.0, 2.0], 3, 1)
        self.assertEqual(2, positive["count days"])
        self.assertEqual(1, negative["count days"])

    def test_mean_confidence_interval_empty_data(self):
        stats = spy._PriceAnalysis__mean_confidence_interval([])
        self.assertEqual([0.0, 0.0, 0.0, 0.0], stats)

    def test_mean_confidence_interval_single_value(self):
        stats = spy._PriceAnalysis__mean_confidence_interval([2.5])
        self.assertEqual([2.5, 0.0, 2.5, 2.5], stats)

    def test_mean_confidence_interval_autocorrelated_series_is_finite(self):
        stats = spy._PriceAnalysis__mean_confidence_interval([1.0, 1.2, 1.1, 1.3, 1.25, 1.35, 1.3, 1.4])
        self.assertEqual(4, len(stats))
        for value in stats:
            self.assertTrue(float(value) == float(value))  # not NaN
        self.assertLessEqual(stats[2], stats[0])
        self.assertGreaterEqual(stats[3], stats[0])

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

        value_keys = [key for key in beyond_bucket.keys() if key not in {"gap", "count days"}]
        self.assertGreater(len(value_keys), 0)
        for key in value_keys:
            self.assertEqual(
                no_data,
                beyond_bucket[key],
                msg=f"Expected no data for {key}, got {beyond_bucket[key]!r}",
            )

    def test_gap_stats_handles_one_sided_opening_data(self):
        analysis = PriceAnalysis("SPY", start, end, dte, "/tmp/")
        analysis._PriceAnalysis__price_history_df = pd.DataFrame(
            {
                "Open wrt close": [0.4, 0.8, 1.2, 2.0],
                "Close wrt close": [0.2, -0.3, 0.4, -0.1],
            }
        )

        analysis._PriceAnalysis__calc_stats_gapup_down()

        self.assertGreater(len(analysis._PriceAnalysis__stats_positive_gap), 0)
        self.assertGreater(len(analysis._PriceAnalysis__stats_negative_gap), 0)
        sample_negative_row = next(iter(analysis._PriceAnalysis__stats_negative_gap.values()))
        sample_positive_row = next(iter(analysis._PriceAnalysis__stats_positive_gap.values()))
        self.assertIn("count days", sample_negative_row)
        self.assertIn("count days", sample_positive_row)
        self.assertGreater(len([k for k in sample_negative_row if k != "gap"]), 0)
        self.assertGreater(len([k for k in sample_positive_row if k != "gap"]), 0)

    def test_gapdown_cumulative_probability_uses_lower_or_equal_threshold(self):
        analysis = PriceAnalysis("SPY", start, end, dte, "/tmp/")
        # All observations are in the first negative-gap bucket (-0.25, 0.0),
        # with close changes chosen to make <= thresholds easy to verify.
        analysis._PriceAnalysis__price_history_df = pd.DataFrame(
            {
                "Open wrt close": [-0.1, -0.1, -0.1, -0.1],
                "Close wrt close": [-2.0, -1.0, 0.0, 1.0],
            }
        )

        analysis._PriceAnalysis__calc_stats_gapup_down()

        first_negative_gap_bucket = analysis._PriceAnalysis__stats_negative_gap["-0.25 %"]
        self.assertEqual(4, int(first_negative_gap_bucket["count days"]))
        # P(close <= -1.0) = 2/4 = 50%
        self.assertEqual(50.0, float(first_negative_gap_bucket["-1.0%"]))
        # P(close <= 0.0) = 3/4 = 75%
        self.assertEqual(75.0, float(first_negative_gap_bucket["0.0%"]))

    def test_data_sanity_check_handles_nan_on_date_column(self):
        analysis = PriceAnalysis("SPY", start, end, dte, "/tmp/")
        analysis._PriceAnalysis__price_history_df = pd.DataFrame(
            {
                "Date": [datetime.date(2024, 1, 5), None],
                "Week number": [1, 1],
                "Year": [2024, 2024],
                "Open": [100.0, 100.0],
                "Close": [101.0, 101.0],
            }
        )

        nan_dates = analysis.data_sanity_check()
        self.assertEqual(1, len(nan_dates))

    def test_trim_gap_table_columns_starts_from_first_negative_with_positive_cdf(self):
        analysis = PriceAnalysis("SPY", start, end, dte, "/tmp/")
        gap_df = pd.DataFrame(
            {
                "gap": ["]0.0; 0.25]%", "]0.25; 0.5]%"],
                "count days": [10, 8],
                "-3.0%": [0.0, 0.0],
                "-2.5%": [0.0, 12.5],
                "-2.0%": [10.0, 25.0],
                "-1.5%": [20.0, 30.0],
                "0.0%": [50.0, 60.0],
            }
        )

        trimmed_df = analysis._PriceAnalysis__trim_gap_table_columns(gap_df)

        self.assertIn("gap", trimmed_df.columns)
        self.assertIn("count days", trimmed_df.columns)
        self.assertNotIn("-3.0%", trimmed_df.columns)
        self.assertEqual("-2.5%", trimmed_df.columns[2])


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

    def test_query_asset_price_uses_iso_year_for_new_year_crossing_week(self):
        asset_df = build_new_year_crossing_week_df()
        analysis = PriceAnalysis(
            "SPY",
            datetime.datetime(2024, 12, 29),
            datetime.datetime(2025, 1, 4),
            3,
            self.output_dir,
            stats_vix=False,
        )

        with patch("helper.data_analysis.yf.Ticker", side_effect=make_ticker_factory(asset_df)):
            analysis.query_asset_price()

        price_df = analysis.get_price_history()
        for _, row in price_df.iterrows():
            iso = pd.Timestamp(row["Date"]).isocalendar()
            self.assertEqual(int(iso.week), int(row["Week number"]))
            self.assertEqual(int(iso.year), int(row["Year"]))

        weekly_frames = analysis._PriceAnalysis__get_weekly_timeframes(min_days=1)
        self.assertEqual(1, len(weekly_frames))
        self.assertEqual(len(asset_df), len(weekly_frames[0]))

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

        with patch("helper.data_analysis.yf.download", return_value=pd.DataFrame()), \
                patch("helper.data_analysis.yf.Ticker", side_effect=make_ticker_factory(pd.DataFrame(), vix_df)):
            analysis.query_vix()

        self.assertEqual([20.0, 20.0, 0.0, 0.0], analysis.get_price_history()["VIX"].tolist())

    def test_normalize_ohlc_dataframe_handles_ticker_first_multiindex(self):
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=True)
        raw_df = pd.DataFrame(
            [[20.1, 20.3, 19.8, 20.0, 1000]],
            index=pd.to_datetime(["2024-01-04"]),
            columns=pd.MultiIndex.from_tuples(
                [
                    ("^VIX", "Open"),
                    ("^VIX", "High"),
                    ("^VIX", "Low"),
                    ("^VIX", "Close"),
                    ("^VIX", "Volume"),
                ]
            ),
        )

        normalized = analysis._PriceAnalysis__normalize_ohlc_dataframe(raw_df)

        self.assertIn("Close", normalized.columns)
        self.assertEqual(20.0, float(normalized["Close"].iloc[0]))

    def test_query_vix_respects_min_supported_start_date(self):
        analysis = PriceAnalysis(
            "SPY",
            datetime.datetime(1989, 1, 1),
            datetime.datetime(1990, 1, 10),
            3,
            self.output_dir,
            stats_vix=True,
        )
        analysis._PriceAnalysis__price_history_df = pd.DataFrame(
            {
                "Date": [
                    datetime.date(1990, 1, 10),
                    datetime.date(1990, 1, 9),
                    datetime.date(1990, 1, 8),
                    datetime.date(1990, 1, 5),
                ],
                "Open": [100.0, 100.0, 100.0, 100.0],
                "Close": [101.0, 101.0, 101.0, 101.0],
            }
        )
        captured = {}

        class CaptureTicker:
            def __init__(self, symbol: str):
                self.symbol = symbol

            def history(self, start=None, end=None):
                captured[self.symbol] = (start, end)
                if self.symbol == "^VIX":
                    return pd.DataFrame(
                        {"Close": [20.0, 21.0]},
                        index=pd.to_datetime(["1990-01-10", "1990-01-09"]),
                    )
                return pd.DataFrame()

        with patch("helper.data_analysis.yf.download", return_value=pd.DataFrame()), \
                patch("helper.data_analysis.yf.Ticker", side_effect=lambda symbol: CaptureTicker(symbol)):
            analysis.query_vix()

        self.assertEqual(datetime.datetime(1990, 1, 2), captured["^VIX"][0])
        self.assertIn("VIX", analysis.get_price_history().columns)

    def test_daily_vix_frequency_is_normalized_across_bins(self):
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=True)
        analysis._PriceAnalysis__price_history_df = pd.DataFrame(
            {
                "VIX": [9.0, 9.5, 11.0, 12.0, 14.0, 42.0, 42.5, 20.0],
                "Close wrt close": [-1.0, 1.0, -2.0, -0.5, 1.2, -0.4, 0.6, 0.0],
            }
        )

        analysis._PriceAnalysis__calc_daily_statistics_vix()

        bins_dict = analysis._PriceAnalysis__dict_daily_change_vix_bins
        negative_frequency_sum = sum(float(v["cumulative negative"]["frequency [%]"]) for v in bins_dict.values())
        positive_frequency_sum = sum(float(v["cumulative positive"]["frequency [%]"]) for v in bins_dict.values())

        self.assertAlmostEqual(100.0, negative_frequency_sum, places=6)
        self.assertAlmostEqual(100.0, positive_frequency_sum, places=6)
        self.assertEqual(2, bins_dict["15"]["cumulative negative"]["count days"])
        self.assertEqual(1, bins_dict["10"]["cumulative negative"]["count days"])
        self.assertEqual(1, bins_dict["40+"]["cumulative negative"]["count days"])

    def test_query_vix_falls_back_to_stooq_when_yahoo_fails(self):
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
        stooq_vix_df = pd.DataFrame(
            {"Close": [20.0]},
            index=pd.to_datetime(["2024-01-04"]),
        )

        with patch("helper.data_analysis.yf.download", side_effect=Exception("yahoo down")), \
                patch("helper.data_analysis.yf.Ticker", side_effect=Exception("yahoo down")), \
                patch("helper.data_analysis.web") as mock_web:
            mock_web.DataReader.return_value = stooq_vix_df
            analysis.query_vix()

        self.assertTrue(mock_web.DataReader.called)
        call_args = mock_web.DataReader.call_args_list[0].args
        self.assertIn(call_args[0], ["^VIX", "VIX"])
        self.assertEqual("stooq", call_args[1])
        self.assertEqual([20.0, 20.0, 0.0, 0.0], analysis.get_price_history()["VIX"].tolist())

    def test_query_vix_falls_back_to_stooq_csv_when_pdr_unavailable(self):
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
        stooq_csv_df = pd.DataFrame(
            {
                "Date": ["2024-01-04", "2024-01-03"],
                "Open": [20.0, 19.5],
                "High": [20.2, 19.8],
                "Low": [19.8, 19.2],
                "Close": [20.0, 19.4],
                "Volume": [100, 100],
            }
        )

        with patch("helper.data_analysis.yf.download", side_effect=Exception("yahoo down")), \
                patch("helper.data_analysis.yf.Ticker", side_effect=Exception("yahoo down")), \
                patch("helper.data_analysis.web", None), \
                patch("helper.data_analysis.pd.read_csv", return_value=stooq_csv_df) as mock_read_csv:
            analysis.query_vix()

        self.assertTrue(mock_read_csv.called)
        self.assertEqual([20.0, 20.0, 19.4, 0.0], analysis.get_price_history()["VIX"].tolist())

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

    def test_calc_change_dte_direction_is_forward_in_time(self):
        analysis = PriceAnalysis("SPY", self.start, self.end, 3, self.output_dir, stats_vix=False)
        dates_desc = [day.date() for day in pd.bdate_range("2024-01-01", periods=8)[::-1]]
        closes_desc = [107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0]
        analysis._PriceAnalysis__price_history_df = pd.DataFrame(
            {
                "Date": dates_desc,
                "Close": closes_desc,
                "Low": [close - 1.0 for close in closes_desc],
                "VIX": [20.0] * len(closes_desc),
            }
        )
        analysis._PriceAnalysis__number_of_trading_days = len(closes_desc)

        change_list_df, _, _ = analysis._PriceAnalysis__calc_change_DTE(3)

        for change in change_list_df["change_list"]:
            self.assertGreater(change, 0.0)

    def test_short_dte_vix_regime_stats_split_counts_correctly(self):
        analysis = PriceAnalysis("SPY", self.start, self.end, 5, self.output_dir, stats_vix=True)
        dates_desc = [day.date() for day in pd.bdate_range("2024-01-01", periods=10)[::-1]]
        analysis._PriceAnalysis__price_history_df = pd.DataFrame(
            {
                "Date": dates_desc,
                "Close": [100.0] * 10,
                "Low": [99.0] * 10,
                "VIX": [0.0, 0.0, 0.0, 0.0, 0.0, 28.0, 26.0, 20.0, 19.0, 10.0],
            }
        )
        analysis._PriceAnalysis__number_of_trading_days = 10

        regime_dfs = analysis._PriceAnalysis__calc_DTE_statistics_by_vix_regime(5, 6)
        low_df = regime_dfs["Low volatility (VIX < 20)"]
        medium_df = regime_dfs["Medium volatility (20 <= VIX < 27)"]
        high_df = regime_dfs["High volatility (VIX >= 27)"]

        self.assertEqual(2, int(low_df.loc["5DTE: positive", "count days"]))
        self.assertEqual(2, int(medium_df.loc["5DTE: positive", "count days"]))
        self.assertEqual(1, int(high_df.loc["5DTE: positive", "count days"]))
        self.assertEqual(0, int(low_df.loc["5DTE: negative", "count days"]))
        self.assertEqual(0, int(medium_df.loc["5DTE: negative", "count days"]))
        self.assertEqual(0, int(high_df.loc["5DTE: negative", "count days"]))

        total_regime_count_days = 0
        for regime_df in regime_dfs.values():
            total_regime_count_days += int(regime_df.loc["5DTE: positive", "count days"])
            total_regime_count_days += int(regime_df.loc["5DTE: negative", "count days"])
        self.assertEqual(5, total_regime_count_days)

    def test_time_direction_consistency_across_metrics(self):
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=False)
        history_df = build_monotonic_bull_history_df(periods=40)
        analysis._PriceAnalysis__price_history_df = history_df.copy()
        analysis._PriceAnalysis__number_of_trading_days = len(history_df)

        analysis._PriceAnalysis__calc_day_change_wrt_previous_day()
        daily_changes = analysis.get_price_history()["Close wrt close"].tolist()[:-1]
        for change in daily_changes:
            self.assertGreater(change, 0.0)

        weekly_changes, _, _ = analysis._PriceAnalysis__calc_weekly_movement()
        self.assertGreater(len(weekly_changes), 0)
        for change in weekly_changes:
            self.assertGreater(change, 0.0)

        friday_to_friday_changes, _, _ = analysis._PriceAnalysis__calc_weekly_friday_to_friday_movement()
        self.assertGreater(len(friday_to_friday_changes), 0)
        for change in friday_to_friday_changes:
            self.assertGreater(change, 0.0)

        for dte_value in [5, 23]:
            dte_change_df, _, _ = analysis._PriceAnalysis__calc_change_DTE(dte_value)
            self.assertGreater(len(dte_change_df["change_list"]), 0)
            for change in dte_change_df["change_list"]:
                self.assertGreater(change, 0.0)

    def test_weekly_plot_hides_date_ticks_but_keeps_period_in_hover(self):
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=False)
        history_df = build_monotonic_bull_history_df(periods=40)
        analysis._PriceAnalysis__price_history_df = history_df.copy()
        analysis._PriceAnalysis__number_of_trading_days = len(history_df)
        analysis._PriceAnalysis__calc_weekly_conditional_statistics()

        fig = analysis._PriceAnalysis__make_plot_weekly_change()
        first_trace = fig.data[0]

        self.assertIsNone(fig.layout.xaxis.ticktext)
        self.assertFalse(fig.layout.xaxis.showticklabels)
        self.assertIn("Period: %{customdata}", first_trace.hovertemplate)
        self.assertGreater(len(first_trace.x), 0)
        self.assertEqual(len(first_trace.x), len(first_trace.customdata))
        self.assertIn("-", first_trace.customdata[0])

    def test_weekly_plot_includes_max_vix_line(self):
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=False)
        history_df = build_monotonic_bull_history_df(periods=40)
        analysis._PriceAnalysis__price_history_df = history_df.copy()
        analysis._PriceAnalysis__number_of_trading_days = len(history_df)
        analysis._PriceAnalysis__calc_weekly_conditional_statistics()

        fig = analysis._PriceAnalysis__make_plot_weekly_change()

        max_vix_trace = next(trace for trace in fig.data if trace.name == "max VIX")
        self.assertEqual("y2", max_vix_trace.yaxis)
        self.assertEqual("max VIX", fig.layout.yaxis2.title.text)
        self.assertIn("Max VIX: %{y:.2f}", max_vix_trace.hovertemplate)
        self.assertGreater(len(max_vix_trace.x), 0)

    def test_monthly_plot_does_not_use_dense_date_tick_labels(self):
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=False)
        analysis._PriceAnalysis__change_list_monthly_dte_for_plot_df = {
            "change_list": [1.0, -0.5, 0.8, -1.2],
            "date range": ["01/01 - 01/02/2024", "02/01 - 02/02/2024", "03/01 - 03/02/2024", "04/01 - 04/02/2024"],
        }

        fig, _ = analysis._PriceAnalysis__make_plot_monthly_change()

        self.assertIsNone(fig.layout.xaxis.ticktext)
        self.assertIsNone(fig.layout.xaxis.tickvals)

    def test_monthly_plot_hover_uses_period_and_orders_oldest_to_newest(self):
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=False)
        analysis._PriceAnalysis__change_list_monthly_dte_for_plot_df = {
            # index 0 = most recent, last index = oldest (production convention)
            "change_list": [1.0, 2.0, 3.0],
            "date range": ["recent", "middle", "oldest"],
        }

        fig, _ = analysis._PriceAnalysis__make_plot_monthly_change()
        positive_trace = fig.data[0]

        self.assertEqual([0, 1, 2], list(positive_trace.x))
        self.assertEqual(["oldest", "middle", "recent"], list(positive_trace.customdata))
        self.assertIn("Period: %{customdata}", positive_trace.hovertemplate)

    def test_monthly_plot_includes_max_vix_line_when_available(self):
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=False)
        analysis._PriceAnalysis__change_list_monthly_dte_for_plot_df = {
            "change_list": [1.0, -0.5, 0.8],
            "date range": ["recent", "middle", "oldest"],
            "max vix": [24.0, 22.0, 20.0],
        }

        fig, _ = analysis._PriceAnalysis__make_plot_monthly_change()

        max_vix_trace = next(trace for trace in fig.data if trace.name == "max VIX")
        self.assertEqual("y2", max_vix_trace.yaxis)
        self.assertEqual("max VIX", fig.layout.yaxis2.title.text)
        self.assertEqual(["oldest", "middle", "recent"], list(max_vix_trace.customdata))
        self.assertIn("Max VIX: %{y:.2f}", max_vix_trace.hovertemplate)

    def test_run_generates_html_report_with_mocked_data(self):
        asset_df = build_asset_history_df()
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=False, do_plot=False)

        with patch("helper.data_analysis.yf.Ticker", side_effect=make_ticker_factory(asset_df)):
            analysis.run()

        self.assertTrue(os.path.exists(analysis.FILENAME))
        with open(analysis.FILENAME, "r") as output_file:
            html = output_file.read()

        self.assertIn("Daily and weekly change stats", html)
        self.assertIn("Gap Up/Down Analysis", html)
        self.assertIn("Daily Change by VIX Level", html)
        self.assertIn("Total count days", html)
        self.assertIn("Daily change (CLOSE with respect to the previous day CLOSE) - total count days:", html)
        self.assertIn("Weekly change (Friday CLOSE with respect to the previous week Friday CLOSE or Monday OPEN)", html)
        self.assertIn("Friday to Friday:", html)
        self.assertIn("Monday to Friday:", html)
        self.assertIn("Note: frequency sums to 100% for Monday to Friday, and 100% for Friday to Friday. "
                      "The two are treated separately.", html)
        self.assertIn("Week if Monday positive/negative sample size:", html)
        self.assertIn("Monday positive:", html)
        self.assertIn("Monday negative:", html)
        self.assertIn("Total count days (positive market opening):", html)
        self.assertIn("Total count days (negative market opening):", html)
        daily_section = html.split(
            "Daily change (CLOSE with respect to the previous day CLOSE) - total count days:"
        )[1].split(
            "Weekly change (Friday CLOSE with respect to the previous week Friday CLOSE or Monday OPEN)"
        )[0]
        self.assertNotIn("Total count days", daily_section)
        monday_conditional_section = html.split(
            "Week if Monday positive/negative sample size:"
        )[1].split(
            "<center><b>Price change analysis with different DTEs</b></center>"
        )[0]
        self.assertNotIn("Total count days", monday_conditional_section)
        gap_section = html.split(
            "<center><b>Gap Up/Down Analysis</b></center>"
        )[1].split(
            "<center><b>Daily Change by VIX Level</b></center>"
        )[0]
        self.assertNotIn("Total count days</td>", gap_section)
        self.assertIn("Asset data source: yahoo", html)
        self.assertIn("VIX data source: disabled", html)
        self.assertEqual("disabled", analysis.get_vix_source())
        self.assertTrue((analysis.get_price_history()["VIX"] == 0.0).all())
        self.assertIn("count days", analysis._PriceAnalysis__daily_change_df.columns)
        self.assertIn("count weeks", analysis._PriceAnalysis__weekly_change_df.columns)
        self.assertIn("count weeks", analysis._PriceAnalysis__weekly_change_monday_conditional_df.columns)
        if analysis._PriceAnalysis__monthly_dte_change_df is not None:
            self.assertIn("count days", analysis._PriceAnalysis__monthly_dte_change_df.columns)
        self.assertEqual(
            0,
            analysis._PriceAnalysis__dict_daily_change_vix_bins["40+"]["cumulative negative"]["count days"],
        )

    def test_run_skips_short_dte_when_trading_days_are_below_week(self):
        asset_df = build_asset_history_df().head(4)
        analysis = PriceAnalysis(
            "SPY",
            datetime.datetime(2024, 1, 1),
            datetime.datetime(2024, 1, 10),
            3,
            self.output_dir,
            stats_vix=False,
            do_plot=False,
        )

        with patch("helper.data_analysis.yf.Ticker", side_effect=make_ticker_factory(asset_df)):
            analysis.run()

        self.assertIsNone(analysis._PriceAnalysis__weekly_short_dte_change_df)

    def test_run_generates_html_report_with_custom_long_dte(self):
        asset_df = build_asset_history_df()
        custom_dte = 10
        analysis = PriceAnalysis("SPY", self.start, self.end, custom_dte, self.output_dir, stats_vix=False, do_plot=False)

        with patch("helper.data_analysis.yf.Ticker", side_effect=make_ticker_factory(asset_df)):
            analysis.run()

        self.assertTrue(os.path.exists(analysis.FILENAME))
        with open(analysis.FILENAME, "r") as output_file:
            html = output_file.read()

        self.assertIn(f"Change in {custom_dte} DTE", html)
        self.assertIn(f"Stats {custom_dte} DTE negative change", html)
        self.assertNotIn("Change in 23 DTE", html)
        expected_last_open_date = analysis.get_price_history()["Date"][custom_dte].strftime('%d/%m/%Y')
        self.assertIn(f"last OPEN {expected_last_open_date}", html)
        monthly_section = html.split(
            f"Change in {custom_dte} DTE"
        )[1].split(
            f"Stats {custom_dte} DTE negative change"
        )[0]
        self.assertNotIn("Total count days", monthly_section)
        self.assertIn("count days", analysis._PriceAnalysis__monthly_dte_change_df.columns)

    def test_run_generates_html_report_with_plots_enabled(self):
        asset_df = build_asset_history_df()
        analysis = PriceAnalysis("SPY", self.start, self.end, 10, self.output_dir, stats_vix=False, do_plot=True)

        with patch("helper.data_analysis.yf.Ticker", side_effect=make_ticker_factory(asset_df)):
            analysis.run()

        self.assertTrue(os.path.exists(analysis.FILENAME))
        with open(analysis.FILENAME, "r") as output_file:
            html = output_file.read()

        self.assertIn("plotly-graph-div", html)
        self.assertIn("Weekly change", html)
        self.assertIn("10 DTE change", html)

    def test_run_tracks_vix_source_in_report_header(self):
        asset_df = build_asset_history_df()
        vix_df = pd.DataFrame(
            {"Close": [20.0, 19.8, 19.5]},
            index=pd.to_datetime(["2024-01-05", "2024-01-04", "2024-01-03"]),
        )
        analysis = PriceAnalysis("SPY", self.start, self.end, 23, self.output_dir, stats_vix=True, do_plot=False)

        with patch("helper.data_analysis.yf.Ticker", side_effect=make_ticker_factory(asset_df, vix_df)), \
                patch("helper.data_analysis.yf.download", return_value=vix_df.copy()):
            analysis.run()

        self.assertTrue(os.path.exists(analysis.FILENAME))
        with open(analysis.FILENAME, "r") as output_file:
            html = output_file.read()

        self.assertIn("Asset data source: yahoo", html)
        self.assertIn("VIX data source: yahoo_download (^VIX)", html)
        self.assertIn("Short DTE change according to VIX regime", html)
        self.assertIn("Low volatility (VIX < 20)", html)
        self.assertIn("Medium volatility (20 <= VIX < 27)", html)
        self.assertIn("High volatility (VIX >= 27)", html)
        self.assertEqual("yahoo_download (^VIX)", analysis.get_vix_source())

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
