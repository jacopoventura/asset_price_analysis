# Copyright (c) 2024 Jacopo Ventura

import datetime
import math
import os
import sys
from pathlib import Path
from urllib.parse import quote

import appdirs as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import t
import streamlit as st
import yfinance as yf

try:
    from pandas_datareader import data as web
except Exception:
    web = None

CACHE_DIR = ".cache"

# Force appdirs to say that the cache dir is .cache
ad.user_cache_dir = lambda *args: CACHE_DIR

# Create the cache dir if it doesn't exist
Path(CACHE_DIR).mkdir(exist_ok=True)


class PriceAnalysis:
    """
    Class to analyze historical pricing data of a specific asset.
    """

    def __init__(self, ticker: str,
                 start: datetime.datetime,
                 end: datetime.datetime,
                 dte_long: int,
                 path_to_report: str,
                 stats_vix: bool = True,
                 do_plot: bool = False):
        """
        Initialize the class PriceAnalysis with ticker, start and end time for the price analysis.
        :param ticker: ticker of the asset class
        :type ticker: str
        :param start: start date for the price analysis
        :type start:datetime
        :param end: end date for the price analysis
        :type end: datetime
        :param dte_long number of trading days of the long option operation
        :type dte_long: int
        :param path_to_report: path where the report is saved
        :type path_to_report: str
        :param stats_vix: query vix data or not
        :type stats_vix: bool
        :param do_plot: plot graphs in the HTML file
        :type do_plot: false
        """

        self.__SOURCE = 'stooq'
        self.__VIX_SOURCE = "disabled" if not stats_vix else "not queried"

        self.__DO_PLOT = do_plot
        self.__STATS_VIX = stats_vix

        self.__WEEK_TRADING_DAYS = 5
        self.__MONTH_TRADING_DAYS = 23
        self.__WEEK_MAX_CHANGE_PCT = 6
        self.__MONTH_MAX_CHANGE_PCT = 12
        self.__NUMBER_WEEKS_PER_YEAR = 52
        self.__STEP = 1  # step to calculate the cumulative distribution
        self.__BINS_DAILY_CHANGE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        self.__BINS_VIX = [vix for vix in range(10, 41, 3)]

        self.__PLOT_COLUMN_WIDTH = 0.75

        self.__STEP_GAP_OPEN = 0.25
        self.__BIN_CLOSE_PCT = 0.5
        self.__MAX_GAP = 2.5

        self.__NO__DATA_INDICATOR = "ND"

        self.__DTE_LONG = dte_long
        self.__ticker = ticker
        self.__date_start = start
        self.__date_end = end
        self.__number_of_days = (end - start).days
        if self.__number_of_days < np.ceil((self.__DTE_LONG / 5) * 7):
            st.error(f'Incorrect input dates. Minimum {self.__DTE_LONG} trading days shall be considered.', icon="🚨")
            sys.exit(1)

        self.__number_of_weeks = math.ceil(self.__number_of_days / 7)
        self.__number_of_trading_days = 0

        self.__PATH_TO_HTML = path_to_report
        if start.year == end.year:
            self.__years_analysis = str(int(start.year))
        else:
            self.__years_analysis = str(int(start.year)) + '-' + str(int(end.year))
        self.FILENAME = self.__PATH_TO_HTML + self.__ticker \
                                              + '_' + self.__date_start.strftime('%d%m%Y') \
                                              + '_to_' \
                                              + self.__date_end.strftime('%d%m%Y') \
                                              + '.html'

        date_start_vix = datetime.datetime(1990, 1, 2)
        if self.__date_start < date_start_vix:
            print("WARNING: start date before first available date for VIX: VIX wil not be queried only from 02/01/1990")
            self.__date_start_vix = date_start_vix
        else:
            self.__date_start_vix = self.__date_start

        self.__price_history_df = None
        self.__daily_change_df = None
        self.__weekly_change_monday_to_friday = None
        self.__weekly_change_friday_to_friday = None
        self.__weekly_change_monday_to_friday_count_weeks = 0
        self.__weekly_change_friday_to_friday_count_weeks = 0
        self.__weekly_change_monday_to_friday_count_days = 0
        self.__weekly_change_friday_to_friday_count_days = 0
        self.__weekly_if_monday_positive_count_weeks = 0
        self.__weekly_if_monday_negative_count_weeks = 0
        self.__weekly_if_monday_positive_count_days = 0
        self.__weekly_if_monday_negative_count_days = 0
        self.__weekly_change_df = None
        self.__weekly_change_monday_conditional_df = None
        self.__weekly_short_dte_change_df = None
        self.__weekly_short_dte_change_vix_regime_dfs = {}
        self.__monthly_dte_change_df = None
        self.__change_list_monthly_dte_for_plot_df = None
        self.__day_gapup_df = None
        self.__day_gapdown_df = None
        self.__stats_positive_gap = {}
        self.__stats_negative_gap = {}

        self.__week_one_weekday_list = []
        self.__years_list = []
        self.__weekly_change_first_day_positive = []
        self.__weekly_change_first_day_negative = []
        self.__weekly_change_first_day_positive_week_count = []
        self.__weekly_change_first_day_negative_week_count = []
        self.__day_positive_close = []
        self.__day_negative_close = []

        self.__dict_daily_change_vix_bins = {}
        for vix in self.__BINS_VIX:
            self.__dict_daily_change_vix_bins[str(vix)] = {"cumulative positive": {}, "cumulative negative": {}}
        self.__dict_daily_change_vix_bins[str(max(self.__BINS_VIX))+"+"] = {"cumulative positive": {}, "cumulative negative": {}}

    def run(self):
        """
        Run the whole analysis and save the final document with all the statistics.
        """

        # Step 1: get historical price data for the selected time period and check data quality
        self.query_asset_price()
        nan_date_list = self.data_sanity_check()
        if nan_date_list:
            print("Historical data have some NaNs")
            print("Check the following dates: ")
            print(nan_date_list)
            return
        if self.__STATS_VIX:
            self.query_vix()
        else:
            self.__price_history_df["VIX"] = 0.0
            self.__VIX_SOURCE = "disabled"

        # Step 2: calculate daily statistics
        self.__calc_daily_statistics()
        self.__calc_daily_statistics_vix()

        # Step 3: calculate weekly statistics
        self.__calc_weekly_statistics()
        self.__calc_weekly_conditional_statistics()
        if self.__number_of_trading_days >= self.__WEEK_TRADING_DAYS:
            self.__weekly_short_dte_change_df = self.__calc_DTE_statistics(self.__WEEK_TRADING_DAYS,
                                                                           self.__WEEK_MAX_CHANGE_PCT)
            self.__weekly_short_dte_change_vix_regime_dfs = self.__calc_DTE_statistics_by_vix_regime(
                self.__WEEK_TRADING_DAYS,
                self.__WEEK_MAX_CHANGE_PCT,
            )

        # Step 4: calculate monthly statistics
        if self.__number_of_trading_days >= self.__DTE_LONG:
            self.__monthly_dte_change_df = self.__calc_DTE_statistics(self.__DTE_LONG,
                                                                      self.__MONTH_MAX_CHANGE_PCT)

        # Step 5: calculate gap-ups and -downs statistics
        self.__calc_stats_gapup_down()

        # Step 5: make HTML report
        self.__write_html()
        print("Report written in: " + self.FILENAME)

    def data_sanity_check(self) -> list:
        """
        Check if any NaN in the queried data.
        :return: list of dates where NaN were found
        :rtype: list
        """
        COLUMNS_TO_CHECK = ["Date", "Week number", "Year", "Open", "Close"]
        for key in COLUMNS_TO_CHECK:
            mask_nan = self.__price_history_df[key].isna()
            if mask_nan.any():
                return self.__price_history_df.loc[mask_nan, "Date"].tolist()
        return []

    def __calc_daily_statistics(self):
        """
        Calculate cumulative probability of positive and negative days.
        """
        self.__calc_day_change_wrt_previous_day()
        self.__split_for_day_performance()
        cumulative_prob_daily_positive_dict = self.__calc_cumulative_probability(self.__day_positive_close)
        cumulative_prob_daily_negative_dict = self.__calc_cumulative_probability(self.__day_negative_close)

        cumulative_prob_daily_positive_dict["frequency [%]"] = \
            100.0 * len(self.__day_positive_close) / (len(self.__price_history_df) - 1)
        cumulative_prob_daily_negative_dict["frequency [%]"] = \
            100.0 * len(self.__day_negative_close) / (len(self.__price_history_df) - 1)

        cumulative_prob_daily_positive_dict["Day"] = "positive day"
        cumulative_prob_daily_negative_dict["Day"] = "negative day"

        # noinspection PyTypeChecker
        self.__daily_change_df = pd.DataFrame.from_dict([cumulative_prob_daily_positive_dict,
                                                        cumulative_prob_daily_negative_dict
                                                         ])
        self.__daily_change_df.set_index("Day", inplace=True)
        self.__daily_change_df.index.name = None

    def __calc_stats_gapup_down(self):
        """
        Calculate the statistics of daily gap-ups and gap-downs.
        :return:
        """
        daily_open_pct = self.__price_history_df["Open wrt close"].values
        daily_close_pct = self.__price_history_df["Close wrt close"].values

        # split between positive and negative openings
        positive_open = []
        close_positive_open = []
        negative_open = []
        close_negative_open = []
        for i, gap in enumerate(daily_open_pct):
            if gap >= 0:
                positive_open.append(gap)
                close_positive_open.append(daily_close_pct[i])
            else:
                negative_open.append(gap)
                close_negative_open.append(daily_close_pct[i])

        # Build close-change bins robustly for one-sided datasets.
        x_cpf_positive_open = [0.0]
        if close_positive_open:
            min_close_positive_open = float(np.min(close_positive_open))
            max_close_positive_open = float(np.max(close_positive_open))
            x_cpf_positive_open = []
            x = math.floor(min_close_positive_open * self.__BIN_CLOSE_PCT) / self.__BIN_CLOSE_PCT + self.__BIN_CLOSE_PCT
            while x < max_close_positive_open:
                x_cpf_positive_open.append(x)
                x += self.__BIN_CLOSE_PCT
            if len(x_cpf_positive_open) == 0 or x_cpf_positive_open[-1] != max_close_positive_open:
                x_cpf_positive_open.append(max_close_positive_open)

        x_cpf_negative_open = [0.0]
        if close_negative_open:
            min_close_negative_open = float(np.min(close_negative_open))
            max_close_negative_open = float(np.max(close_negative_open))
            x_cpf_negative_open = []
            x = math.floor(min_close_negative_open * self.__BIN_CLOSE_PCT) / self.__BIN_CLOSE_PCT + self.__BIN_CLOSE_PCT
            while x < max_close_negative_open:
                x_cpf_negative_open.append(x)
                x += self.__BIN_CLOSE_PCT
            if len(x_cpf_negative_open) == 0 or x_cpf_negative_open[-1] != max_close_negative_open:
                x_cpf_negative_open.append(max_close_negative_open)

        gap_positive_list = []
        gap_negative_list = []
        x = self.__STEP_GAP_OPEN
        while x <= self.__MAX_GAP:
            gap_positive_list.append(x)
            gap_negative_list.append(-x)
            x += self.__STEP_GAP_OPEN

        # gap up
        for gap in gap_positive_list:
            close_list = [daily_close_pct[i] for i in range(len(daily_close_pct)) if ((daily_open_pct[i] <= gap) and (daily_open_pct[i] > (
                    gap - self.__STEP_GAP_OPEN)))]
            key = "]" + str(gap - self.__STEP_GAP_OPEN) + "; " + str(gap) + "]%"
            self.__stats_positive_gap[str(gap) + " %"] = {"gap": key, "count days": len(close_list)}
            if close_list:
                cpf = self.__calc_cpf(close_list, x_cpf_positive_open)
            else:
                cpf = [self.__NO__DATA_INDICATOR] * len(x_cpf_positive_open)
            for idx, close_pct in enumerate(x_cpf_positive_open):
                self.__stats_positive_gap[str(gap) + " %"][str(int(close_pct * 10) / 10) + "%"] = cpf[idx]

        # above the max gap considered in the list
        close_list = [daily_close_pct[i] for i in range(len(daily_close_pct)) if daily_open_pct[i] > gap_positive_list[-1]]
        gap = gap_positive_list[-1]
        key = ">" + str(gap) + " %"
        self.__stats_positive_gap[">+" + str(gap) + " %"] = {"gap": key, "count days": len(close_list)}
        if close_list:
            cpf = self.__calc_cpf(close_list, x_cpf_positive_open)
        else:
            cpf = [self.__NO__DATA_INDICATOR] * len(x_cpf_positive_open)
        for idx, close_pct in enumerate(x_cpf_positive_open):
            self.__stats_positive_gap[">+" + str(gap) + " %"][str(int(close_pct * 10) / 10) + "%"] = cpf[idx]

        # all positive gap-ups
        close_list = [daily_close_pct[i] for i in range(len(daily_close_pct)) if daily_open_pct[i] > 0]
        key = "positive open %"
        self.__stats_positive_gap[">0 %"] = {"gap": key, "count days": len(close_list)}
        if close_list:
            cpf = self.__calc_cpf(close_list, x_cpf_positive_open)
        else:
            cpf = [self.__NO__DATA_INDICATOR] * len(x_cpf_positive_open)
        for idx, close_pct in enumerate(x_cpf_positive_open):
            self.__stats_positive_gap[">0 %"][str(int(close_pct * 10) / 10) + "%"] = cpf[idx]

        # gap down
        for gap in gap_negative_list:
            close_list = [daily_close_pct[i] for i in range(len(daily_close_pct)) if ((daily_open_pct[i] >= gap) and (daily_open_pct[i] < (
                    gap + self.__STEP_GAP_OPEN)))]
            key = "[" + str(gap) + "; " + str(gap - self.__STEP_GAP_OPEN) + "[%"
            self.__stats_negative_gap[str(gap) + " %"] = {"gap": key, "count days": len(close_list)}
            if close_list:
                cpf = self.__calc_cpf(close_list, x_cpf_negative_open)
            else:
                cpf = [self.__NO__DATA_INDICATOR]*len(x_cpf_negative_open)
            for idx, close_pct in enumerate(x_cpf_negative_open):
                self.__stats_negative_gap[str(gap) + " %"][str(int(close_pct * 10) / 10) + "%"] = cpf[idx]

        # above the max gap considered in the list
        close_list = [daily_close_pct[i] for i in range(len(daily_close_pct)) if daily_open_pct[i] < gap_negative_list[-1]]
        gap = gap_negative_list[-1]
        key = ">" + str(gap) + " %"
        self.__stats_negative_gap[">+" + str(gap) + " %"] = {"gap": key, "count days": len(close_list)}
        if close_list:
            cpf = self.__calc_cpf(close_list, x_cpf_negative_open)
        else:
            cpf = [self.__NO__DATA_INDICATOR] * len(x_cpf_negative_open)
        for idx, close_pct in enumerate(x_cpf_negative_open):
            self.__stats_negative_gap[">+" + str(gap) + " %"][str(int(close_pct * 10) / 10) + "%"] = cpf[idx]

        # all negative gap-ups
        close_list = [daily_close_pct[i] for i in range(len(daily_close_pct)) if daily_open_pct[i] < 0]
        key = "negative open %"
        self.__stats_negative_gap[">0 %"] = {"gap": key, "count days": len(close_list)}
        if close_list:
            cpf = self.__calc_cpf(close_list, x_cpf_negative_open)
        else:
            cpf = [self.__NO__DATA_INDICATOR] * len(x_cpf_negative_open)
        for idx, close_pct in enumerate(x_cpf_negative_open):
            self.__stats_negative_gap[">0 %"][str(int(close_pct * 10) / 10) + "%"] = cpf[idx]

    @staticmethod
    def __calc_cpf(data: list, x_cpf: list) -> list:
        """
        Calculate the cumulative probability function.
        :param data: input data list on which the cpf is calculated
        :type data: list
        :param x_cpf: bins to calculate the cpf
        :type x_cpf: list
        """

        cdf = []
        n = float(len(data))
        if n > 0:
            for x in x_cpf:
                cdf.append(100. * sum(i <= x for i in data) / n)
        return cdf

    def __calc_cumulative_probability(self, input_data: list) -> dict:
        """
        Calculate the cumulative probability of the input data.
        :param input_data: data for which the cumulative probability is calculated
        :type input_data: list
        :return: cumulative_prob_list: list of the cumulative probability
        :rtype: list
        """
        data = np.sort(input_data, kind="stable")

        dict_cumulative_dist = {"frequency [%]": 0, "count days": 0}
        num_days = len(data)
        dict_cumulative_dist["count days"] = num_days
        if num_days == 0:
            for pct in self.__BINS_DAILY_CHANGE:
                dict_cumulative_dist[str(int(pct * 10) / 10) + "% change"] = 0.0
            return dict_cumulative_dist

        if data[0] >= 0:  # positive data if min >=0, otherwise negative data
            for pct in self.__BINS_DAILY_CHANGE:
                dict_cumulative_dist[str(int(pct*10)/10) + "% change"] = \
                    100.0 * len([j for j in data if j <= pct]) / num_days
        else:
            for pct in self.__BINS_DAILY_CHANGE:
                dict_cumulative_dist[str(int(pct*10)/10) + "% change"] = \
                    100.0 * len([j for j in data if j >= -pct]) / num_days

        return dict_cumulative_dist

    def update_analysis_period(self, start: datetime.datetime, end: datetime.datetime):
        """
        Update the time period for the analysis.
        :param start: start date for the price analysis
        :type start:datetime
        :param end: end date for the price analysis
        :type end: datetime
        """

        self.__date_start = start
        self.__date_end = end
        self.__number_of_weeks = math.ceil((end - start).days / 7)

        if start.year == end.year:
            self.__years_analysis = str(int(start.year))
        else:
            self.__years_analysis = str(int(start.year)) + '-' + str(int(end.year))
        self.FILENAME = self.__PATH_TO_HTML + self.__ticker + '_' \
                                              + self.__years_analysis \
                                              + '_update' + self.__date_end.strftime('%d%m%Y') \
                                              + '.html'

    def query_asset_price(self):
        """
        Query data of the ticker for the input timerange from the source database.
        """

        self.__price_history_df = pd.DataFrame()
        source_used = "yahoo"

        try:
            ticker_obj = yf.Ticker(self.__ticker)
            self.__price_history_df = ticker_obj.history(start=self.__date_start, end=self.__date_end)
        except Exception as error:
            print(f"WARNING: Yahoo Finance query failed for {self.__ticker}: {error}")

        if self.__price_history_df.empty and web is not None:
            try:
                self.__price_history_df = web.DataReader(
                    self.__ticker,
                    "stooq",
                    self.__date_start,
                    self.__date_end
                )
                source_used = "stooq"
                print(f"INFO: using stooq fallback source for {self.__ticker}")
            except Exception as error:
                print(f"WARNING: stooq fallback query failed for {self.__ticker}: {error}")

        if self.__price_history_df.empty:
            st.error('Could not query price data. Please check that the ticker is correct and run the app again.', icon="🚨")
            sys.exit(1)

        self.__SOURCE = source_used

        # except ValueError:
        #    st.error('Cannot query historical data')
        #    sys.exit(1)  # stop the main function with exit code 1

        # Remove hour from index column (date)
        self.__price_history_df["Date"] = [d.date() for d in self.__price_history_df.index.to_list()]
        self.__price_history_df.set_index("Date", inplace=True)

        # sort by date (first row the most recent date)
        date_list = self.__price_history_df.index
        # a date in the past is always smaller than a more recent day
        if date_list[0] < date_list[1]:
            self.__price_history_df.sort_index(inplace=True, ascending=False)

        weekday_list = []
        weeknumber_list = []
        years_list = []
        for index, _ in self.__price_history_df.iterrows():
            d = pd.to_datetime(index)
            iso = d.isocalendar()
            weekday_list.append(d.weekday())
            weeknumber_list.append(iso[1])
            years_list.append(iso[0])
        self.__price_history_df.insert(0, "Weekday", weekday_list)
        self.__price_history_df.insert(1, "Week number", weeknumber_list)
        self.__price_history_df.insert(2, "Year", years_list)
        self.__price_history_df.insert(len(self.__price_history_df.keys()), "Open wrt close", 0)
        self.__price_history_df.insert(len(self.__price_history_df.keys()), "Close wrt close", 0)
        self.__price_history_df = self.__price_history_df.reset_index(level=0)
        self.__years_list = list(set(years_list))  # get unique years
        self.__number_of_trading_days = len(weekday_list)

    def query_vix(self):
        """
        Query data of the VIX for the input timerange from the source database and add it to the main dataframe.
        Download vix data from yahoo (data available from 02.01.1990).
        database:
        finance.yahoo.com/quote/%5EVIX/history?period1=631238400&period2=1689206400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
        """

        self.__VIX_SOURCE = "none"

        if self.__date_end <= self.__date_start_vix:
            print("WARNING: requested VIX period ends before 02/01/1990. VIX stats set to 0.")
            self.__price_history_df["VIX"] = 0.0
            return

        vix_history_df = self.__query_vix_yahoo()
        if vix_history_df.empty:
            vix_history_df = self.__query_vix_stooq()

        if vix_history_df.empty:
            print('WARNING: empty VIX dataset returned from all sources. VIX stats set to 0.')
            self.__price_history_df["VIX"] = 0.0
            return

        vix_close_series = pd.to_numeric(vix_history_df["Close"], errors="coerce")
        vix_index_date = [d.date() for d in pd.to_datetime(vix_history_df.index).to_list()]
        vix_series = pd.Series(vix_close_series.values, index=vix_index_date)
        vix_series = vix_series[~vix_series.index.duplicated(keep="last")]

        self.__price_history_df["VIX"] = self.__price_history_df["Date"].map(vix_series).fillna(0.0)
        self.__price_history_df["VIX"] = self.__price_history_df["VIX"].replace(0.0, np.nan).bfill().fillna(0.0)

    @staticmethod
    def __normalize_ohlc_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
        if input_df.empty:
            return pd.DataFrame()

        output_df = input_df.copy()
        if isinstance(output_df.columns, pd.MultiIndex):
            expected_ohlc_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            level_0_names = [str(col[0]) if isinstance(col, tuple) and len(col) > 0 else str(col) for col in output_df.columns]
            level_1_names = [str(col[1]) if isinstance(col, tuple) and len(col) > 1 else "" for col in output_df.columns]
            level_0_hits = sum(name in expected_ohlc_names for name in level_0_names)
            level_1_hits = sum(name in expected_ohlc_names for name in level_1_names)

            # yfinance can return either (field, ticker) or (ticker, field).
            if level_1_hits > level_0_hits:
                output_df.columns = [col[1] if isinstance(col, tuple) and len(col) > 1 else col for col in output_df.columns]
            else:
                output_df.columns = [col[0] if isinstance(col, tuple) and len(col) > 0 else col for col in output_df.columns]

        rename_map = {
            "close": "Close",
            "adj close": "Adj Close",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "volume": "Volume",
        }
        output_df.rename(
            columns={column: rename_map.get(str(column).lower(), column) for column in output_df.columns},
            inplace=True,
        )

        if "Close" not in output_df.columns and "Adj Close" in output_df.columns:
            output_df["Close"] = output_df["Adj Close"]

        if "Close" not in output_df.columns:
            print(f"WARNING: queried dataframe has no Close column. Available columns: {list(output_df.columns)}")
            return pd.DataFrame()

        output_df = output_df.loc[:, ~output_df.columns.duplicated(keep="last")]
        output_df = output_df.loc[~output_df.index.duplicated(keep="last")]
        return output_df

    def __query_vix_yahoo(self) -> pd.DataFrame:
        vix_history_df = pd.DataFrame()
        query_methods = [
            ("yahoo_download (^VIX)", lambda: yf.download(
                "^VIX",
                start=self.__date_start_vix,
                end=self.__date_end,
                group_by="column",
                progress=False,
                auto_adjust=False,
                threads=False,
            )),
            ("yahoo_history (^VIX)", lambda: yf.Ticker("^VIX").history(start=self.__date_start_vix, end=self.__date_end)),
        ]

        for source_label, query_method in query_methods:
            try:
                vix_history_df = self.__normalize_ohlc_dataframe(query_method())
                if not vix_history_df.empty:
                    self.__VIX_SOURCE = source_label
                    return vix_history_df
            except Exception as error:
                print(f"WARNING: cannot query historical data of VIX from Yahoo: {error}")
        return pd.DataFrame()

    def __query_vix_stooq(self) -> pd.DataFrame:
        stooq_symbols = ["^VIX", "VIX"]

        if web is not None:
            for vix_symbol in stooq_symbols:
                try:
                    vix_history_df = web.DataReader(vix_symbol, "stooq", self.__date_start_vix, self.__date_end)
                    vix_history_df = self.__normalize_ohlc_dataframe(vix_history_df)
                    if not vix_history_df.empty:
                        self.__VIX_SOURCE = f"stooq_pdr ({vix_symbol})"
                        print(f"INFO: using stooq fallback source for {vix_symbol}")
                        return vix_history_df
                except Exception as error:
                    print(f"WARNING: stooq fallback query failed for {vix_symbol}: {error}")

        for vix_symbol in ["^VIX", "^vix", "VIX", "vix"]:
            try:
                stooq_url = (
                    "https://stooq.com/q/d/l/"
                    f"?s={quote(vix_symbol)}"
                    f"&i=d&d1={self.__date_start_vix.strftime('%Y%m%d')}"
                    f"&d2={self.__date_end.strftime('%Y%m%d')}"
                )
                vix_history_df = pd.read_csv(stooq_url)
                vix_history_df = self.__normalize_ohlc_dataframe(vix_history_df)
                if vix_history_df.empty:
                    continue
                if "Date" in vix_history_df.columns:
                    vix_history_df["Date"] = pd.to_datetime(vix_history_df["Date"])
                    vix_history_df.set_index("Date", inplace=True)
                if not vix_history_df.empty:
                    self.__VIX_SOURCE = f"stooq_csv ({vix_symbol})"
                    print(f"INFO: using stooq CSV fallback source for {vix_symbol}")
                    return vix_history_df
            except Exception as error:
                print(f"WARNING: stooq CSV fallback query failed for {vix_symbol}: {error}")

        return pd.DataFrame()

    def __calc_daily_statistics_vix(self):
        """
        Calculate the cumulative probability of daily change (close to close) according to the VIX levels.
        Data in each bin interval are selected such that vix>vix_min & vix<=vix_max.
        :return: None
        """

        # Initialize all bins with empty cumulative distributions, so sample-size
        # information is available even when a bin has no observations.
        for key in self.__dict_daily_change_vix_bins.keys():
            self.__dict_daily_change_vix_bins[key]["cumulative negative"] = self.__calc_cumulative_probability([])
            self.__dict_daily_change_vix_bins[key]["cumulative positive"] = self.__calc_cumulative_probability([])

        vix_changes_by_bin = {}
        vix_labels_by_bin = {}

        min_vix = min(self.__BINS_VIX)
        max_vix = max(self.__BINS_VIX)

        min_vix_key = str(min_vix)
        filtered_df = self.__price_history_df.loc[self.__price_history_df['VIX'] <= min_vix]
        vix_changes_by_bin[min_vix_key] = {
            "positive": filtered_df.loc[filtered_df["Close wrt close"] > 0]["Close wrt close"].tolist(),
            "negative": filtered_df.loc[filtered_df["Close wrt close"] < 0]["Close wrt close"].tolist(),
        }
        vix_labels_by_bin[min_vix_key] = min_vix_key + "]"

        max_vix_key = str(max_vix) + "+"
        filtered_df = self.__price_history_df.loc[self.__price_history_df['VIX'] > max_vix]
        vix_changes_by_bin[max_vix_key] = {
            "positive": filtered_df.loc[filtered_df["Close wrt close"] > 0]["Close wrt close"].tolist(),
            "negative": filtered_df.loc[filtered_df["Close wrt close"] < 0]["Close wrt close"].tolist(),
        }
        vix_labels_by_bin[max_vix_key] = "]" + max_vix_key

        # Check the vix intervals
        for idx in range(len(self.__BINS_VIX) - 1):
            vix_min = self.__BINS_VIX[idx]
            vix_max = self.__BINS_VIX[idx + 1]
            vix_key = str(vix_max)
            filtered_df = self.__price_history_df.loc[
                (self.__price_history_df['VIX'] > vix_min) & (self.__price_history_df['VIX'] <= vix_max)
            ]
            vix_changes_by_bin[vix_key] = {
                "positive": filtered_df.loc[filtered_df["Close wrt close"] > 0]["Close wrt close"].tolist(),
                "negative": filtered_df.loc[filtered_df["Close wrt close"] < 0]["Close wrt close"].tolist(),
            }
            vix_labels_by_bin[vix_key] = "]" + str(vix_min) + "; " + str(vix_max) + "]"

        total_positive_days = sum(len(v["positive"]) for v in vix_changes_by_bin.values())
        total_negative_days = sum(len(v["negative"]) for v in vix_changes_by_bin.values())

        for vix_key in self.__dict_daily_change_vix_bins.keys():
            positive_daily_change = vix_changes_by_bin[vix_key]["positive"]
            negative_daily_change = vix_changes_by_bin[vix_key]["negative"]

            if negative_daily_change:
                cumulative_negative = self.__calc_cumulative_probability(negative_daily_change)
                cumulative_negative["frequency [%]"] = 100.0 * len(negative_daily_change) / total_negative_days
                cumulative_negative["count days"] = len(negative_daily_change)
                self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"] = cumulative_negative

            if positive_daily_change:
                cumulative_positive = self.__calc_cumulative_probability(positive_daily_change)
                cumulative_positive["frequency [%]"] = 100.0 * len(positive_daily_change) / total_positive_days
                cumulative_positive["count days"] = len(positive_daily_change)
                self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"] = cumulative_positive

            self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"]["VIX"] = vix_labels_by_bin[vix_key]
            self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"]["VIX"] = vix_labels_by_bin[vix_key]

    @staticmethod
    def __append_total_count_days_row(df: pd.DataFrame) -> pd.DataFrame:
        """
        Append a summary row with the total count of days when the table contains
        a "count days" column.
        """

        if "count days" not in df.columns:
            return df

        output_df = df.copy()
        total_count_days = pd.to_numeric(output_df["count days"], errors="coerce").fillna(0).sum()
        total_row = {column: 0 for column in output_df.columns}
        total_row["count days"] = total_count_days

        row_label = "Total count days"
        while row_label in output_df.index:
            row_label += " "
        output_df.loc[row_label] = total_row
        return output_df

    @staticmethod
    def __get_total_count_days(df: pd.DataFrame) -> float:
        """
        Return total count days if present in the table, else 0.
        """

        if "count days" not in df.columns:
            return 0
        return pd.to_numeric(df["count days"], errors="coerce").fillna(0).sum()

    @staticmethod
    def __parse_pct_column(column_name: str) -> float | None:
        """
        Parse a percentage-like column name (e.g. "-1.5%") to float.
        """

        text = str(column_name).strip()
        if not text.endswith("%"):
            return None
        try:
            return float(text[:-1])
        except ValueError:
            return None

    def __trim_gap_table_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trim leading negative-change columns that have 0 cumulative probability
        for all rows. Keep table starting from the first negative column where at
        least one row has cumulative probability > 0.
        """

        if df.empty:
            return df

        negative_pct_columns = [
            column for column in df.columns
            if (self.__parse_pct_column(column) is not None and self.__parse_pct_column(column) < 0)
        ]
        if not negative_pct_columns:
            return df

        first_negative_column_with_positive_cdf = None
        for column in negative_pct_columns:
            values = pd.to_numeric(df[column], errors="coerce").fillna(0)
            if (values > 0).any():
                first_negative_column_with_positive_cdf = column
                break

        if first_negative_column_with_positive_cdf is None:
            return df

        start_idx = list(df.columns).index(first_negative_column_with_positive_cdf)
        keep_columns = []
        for fixed_column in ["gap", "count days"]:
            if fixed_column in df.columns:
                keep_columns.append(fixed_column)
        for column in list(df.columns)[start_idx:]:
            if self.__parse_pct_column(column) is not None:
                keep_columns.append(column)

        return df[keep_columns]

    def __write_html(self):
        """
        Write output file with all the statistics.
        """

        try:
            def df_to_html_1_decimal(df: pd.DataFrame, include_total_row: bool = True) -> str:
                output_df = self.__append_total_count_days_row(df) if include_total_row else df
                return output_df.to_html(float_format=lambda x: f"{x:.1f}").replace('<td>', '<td align="center">')

            with open(os.path.expanduser(self.FILENAME), 'w') as fo:
                fo.write("<html>\n<head>\n<title> \nOutput Data in an HTML file \
                          </title>\n</head> <body><h1><center>" + self.__ticker + "</center></h1>\n</body></html>")
                fo.write("Statistical analysis " + self.__ticker + " " + self.__years_analysis)
                fo.write("<br/>Time period: " + self.__date_start.strftime('%d/%m/%Y'))
                fo.write(" to " + self.__date_end.strftime('%d/%m/%Y'))
                fo.write(" (" + str(int(self.__number_of_weeks)) + " weeks)")
                fo.write("<br/>Number of trading days analyzed: " + str(self.__number_of_trading_days))
                fo.write("<br/>Asset data source: " + self.__SOURCE)
                fo.write("<br/>VIX data source: " + self.__VIX_SOURCE)
                fo.write("<br/>Documented created on: " + datetime.datetime.today().strftime('%d/%m/%Y'))
                fo.write('<br/>' + "Tables contain the <u>cumulative probability</u> of change.")

                # ================================= Daily and Weekly STATS ===================================
                fo.write('<br/><br/>')
                fo.write("<center><b>Daily and weekly change stats</b></center>")
                fo.write(
                    "<br/>The tables in this sections contain the <b>cumulative probability</b> "
                    "of the change in price up to a certain level (column).")
                daily_total_count_days = int(self.__get_total_count_days(self.__daily_change_df))
                fo.write('<br/>' + '<br/>' + "Daily change (CLOSE with respect to the previous day CLOSE)"
                         + " - total count days: " + str(daily_total_count_days))
                fo.write('<br/>')
                fo.write(df_to_html_1_decimal(self.__daily_change_df, include_total_row=False))
                friday_to_friday_count_weeks = int(self.__weekly_change_friday_to_friday_count_weeks)
                monday_to_friday_count_weeks = int(self.__weekly_change_monday_to_friday_count_weeks)
                friday_to_friday_days = int(self.__weekly_change_friday_to_friday_count_days)
                monday_to_friday_days = int(self.__weekly_change_monday_to_friday_count_days)
                fo.write('<br/>' + '<br/>' + "Weekly change (Friday CLOSE with respect to the previous week Friday CLOSE or Monday OPEN)"
                         + "<br/>Friday to Friday: " + str(friday_to_friday_count_weeks) + " weeks ("
                         + str(friday_to_friday_days) + " days)"
                         + "<br/>Monday to Friday: " + str(monday_to_friday_count_weeks) + " weeks ("
                         + str(monday_to_friday_days) + " days)")
                fo.write('<br/>')
                fo.write(df_to_html_1_decimal(self.__weekly_change_df))
                fo.write("<br/>Note: frequency sums to 100% for Monday to Friday, and 100% for Friday to Friday. "
                         "The two are treated separately.")
                fo.write('<br/>')
                monday_positive_count_weeks = int(self.__weekly_if_monday_positive_count_weeks)
                monday_negative_count_weeks = int(self.__weekly_if_monday_negative_count_weeks)
                monday_positive_count_days = int(self.__weekly_if_monday_positive_count_days)
                monday_negative_count_days = int(self.__weekly_if_monday_negative_count_days)
                monday_total_count_weeks = monday_positive_count_weeks + monday_negative_count_weeks
                monday_total_count_days = monday_positive_count_days + monday_negative_count_days
                fo.write("<br/>Week if Monday positive/negative sample size:")
                fo.write("<br/>Monday positive: " + str(monday_positive_count_weeks) + " weeks ("
                         + str(monday_positive_count_days) + " days)")
                fo.write("<br/>Monday negative: " + str(monday_negative_count_weeks) + " weeks ("
                         + str(monday_negative_count_days) + " days)")
                fo.write("<br/>Total: " + str(monday_total_count_weeks) + " weeks ("
                         + str(monday_total_count_days) + " days)")
                fo.write('<br/>')
                fo.write(df_to_html_1_decimal(self.__weekly_change_monday_conditional_df, include_total_row=False))
                fo.write('<br/>')

                # ================================= DTE STATS ===================================
                fo.write('<br/>')
                fo.write("<center><b>Price change analysis with different DTEs</b></center>")
                fo.write('<br/>')
                if self.__weekly_short_dte_change_df is not None:
                    fo.write('<br/>')
                    fo.write("Short DTE: position opened at any day's close and closed at the DTE close")
                    fo.write(df_to_html_1_decimal(self.__weekly_short_dte_change_df))
                    if self.__weekly_short_dte_change_vix_regime_dfs:
                        total_regime_count_days = 0
                        for regime_df in self.__weekly_short_dte_change_vix_regime_dfs.values():
                            total_regime_count_days += int(self.__get_total_count_days(regime_df))
                        fo.write('<br/><br/>')
                        fo.write("Short DTE change according to VIX regime (VIX at position opening)")
                        fo.write("<br/>Total count days: " + str(total_regime_count_days))
                        for regime, regime_df in self.__weekly_short_dte_change_vix_regime_dfs.items():
                            regime_total_count_days = int(self.__get_total_count_days(regime_df))
                            fo.write('<br/><br/><b>' + regime + "</b> - total count days: "
                                     + str(regime_total_count_days))
                            fo.write(df_to_html_1_decimal(regime_df, include_total_row=False))

                # monthly dte
                figure_dte_change = None
                if self.__monthly_dte_change_df is not None:
                    last_open_idx = min(self.__DTE_LONG, self.__number_of_trading_days - 1)
                    last_open_date = self.__price_history_df["Date"][last_open_idx].strftime('%d/%m/%Y')
                    fo.write('<br/>' + '<br/>' + "Change in " + str(self.__DTE_LONG))
                    fo.write(" DTE (effective trading days, (open the position every day close and close at the DTE closing price)")
                    fo.write("<br>" + str(self.__number_of_trading_days) + " analyzed days (last OPEN ")
                    fo.write(last_open_date + ")")
                    fo.write('<br/>')
                    fo.write(df_to_html_1_decimal(self.__monthly_dte_change_df, include_total_row=False))
                    figure_dte_change, negative_change_stats = self.__make_plot_monthly_change()
                    fo.write('Stats ' + str(self.__DTE_LONG) + ' DTE negative change:')
                    fo.write('<br/>')
                    fo.write('Average: ' + '{:.1f}'.format(negative_change_stats[0]) +
                             '%, confidence interval: [' + '{:.1f}'.format(negative_change_stats[2]) + '; ' +
                             '{:.1f}'.format(negative_change_stats[3]) + ']')
                    fo.write('<br/>')
                    fo.write('Standard deviation: ' + '{:.1f}'.format(negative_change_stats[1]) + '%')
                if self.__DO_PLOT:
                    if self.__number_of_weeks > 0:
                        fo.write('<br/>')
                        figure_weekly_change = self.__make_plot_weekly_change()
                        fo.write((figure_weekly_change.to_html(full_html=False, include_plotlyjs='cdn')))
                    if figure_dte_change is not None:
                        fo.write('<br/>')
                        fo.write((figure_dte_change.to_html(full_html=False, include_plotlyjs='cdn')))

                # ================================= GAP UP / DOWN STATS ===================================
                fo.write('<br/><br/><br/>')
                fo.write("<center><b>Open gap-up / down analysis</b></center>")
                fo.write('<br/>')
                fo.write("<br/><b>Cumulative probability of the daily change</b> when a <u>positive market opening</u> occurs:")
                fo.write('<br/><br/>')
                gap_up_total_count_days = int(self.__stats_positive_gap[">0 %"]["count days"]) \
                    if ">0 %" in self.__stats_positive_gap else 0
                fo.write("<br/>Total count days (positive market opening): " + str(gap_up_total_count_days))
                gap_up_df = pd.DataFrame([self.__stats_positive_gap[i] for i in self.__stats_positive_gap.keys()])
                gap_up_df = self.__trim_gap_table_columns(gap_up_df)
                fo.write(df_to_html_1_decimal(gap_up_df, include_total_row=False))
                fo.write("<b>HOW TO USE THE TABLE:</b>")
                fo.write("<br/> - row index: range of the opening gap-up")
                fo.write("<br/> - column: daily change [%] (close with respect to the previous day's close)")
                fo.write("<br/> - cell: <u>cumulative probability [%]</u> that the close is <b>lower or equal</b> the change in the column header")
                fo.write("<br> - USAGE: observe the open gap. Choose the daily sell put strike based on the close pct with the lowest probability.")

                fo.write('<br/><br/>')
                fo.write("<br/><b>Cumulative probability of the daily change</b> when a <u>negative market opening</u> occurs:")
                fo.write('<br/><br/>')
                gap_down_total_count_days = int(self.__stats_negative_gap[">0 %"]["count days"]) \
                    if ">0 %" in self.__stats_negative_gap else 0
                fo.write("<br/>Total count days (negative market opening): " + str(gap_down_total_count_days))
                gap_down_df = pd.DataFrame([self.__stats_negative_gap[i] for i in self.__stats_negative_gap.keys()])
                gap_down_df = self.__trim_gap_table_columns(gap_down_df)
                fo.write(df_to_html_1_decimal(gap_down_df, include_total_row=False))
                fo.write("<b>HOW TO USE THE TABLE:</b>")
                fo.write("<br/> - row index: range of the opening gap-down")
                fo.write("<br/> - column: daily change [%] (close with respect to the previous day's close)")
                fo.write("<br/> - cell: <u>cumulative probability [%]</u> that the close is <b>lower or equal</b> the change in the column header")
                fo.write("<br> - USAGE: observe the open gap. Choose the daily sell put strike based on the close pct with the lowest probability.")

                # ================================= DAILY CHANGE VS. VIX STATS ===================================
                fo.write('<br/><br/><br/>')
                fo.write("<center><b>Daily change according to VIX</b></center>")
                fo.write('<br/>')
                negative_day_vix_df = pd.DataFrame([self.__dict_daily_change_vix_bins[i]["cumulative negative"] for i in
                                                    self.__dict_daily_change_vix_bins.keys()])
                negative_day_vix_df.set_index("VIX", inplace=True)
                # negative_day_vix_df.index.name = None
                negative_day_vix_df = negative_day_vix_df.fillna(0)
                positive_day_vix_df = pd.DataFrame([self.__dict_daily_change_vix_bins[i]["cumulative positive"] for i in
                                                    self.__dict_daily_change_vix_bins.keys()])
                positive_day_vix_df.set_index("VIX", inplace=True)
                # positive_day_vix_df.index.name = None
                positive_day_vix_df = positive_day_vix_df.fillna(0)
                negative_vix_count_days = int(self.__get_total_count_days(negative_day_vix_df))
                positive_vix_count_days = int(self.__get_total_count_days(positive_day_vix_df))
                total_vix_count_days = negative_vix_count_days + positive_vix_count_days
                fo.write("<br/>Total count days across the two VIX tables: " + str(total_vix_count_days))
                fo.write("<br/><b>Cumulative probability</b> of the <b>daily NEGATIVE change</b> according to the <u>vix level</u>:")
                fo.write("<br/>Total count days (daily NEGATIVE change): " + str(negative_vix_count_days))
                fo.write(df_to_html_1_decimal(negative_day_vix_df, include_total_row=False))
                fo.write("<b>HOW TO USE THE TABLE:</b>")
                fo.write("<br/> - row index: range of the VIX")
                fo.write("<br/> - column: daily change [%] (close with respect to the previous day's close)")
                fo.write("<br/> - cell: <u>cumulative probability [%]</u> that the close is <b>lower or equal</b> the change in the column header")
                fo.write('<br/>')
                fo.write("<br/><b>Cumulative probability</b> of the <b>daily POSITIVE change</b> according to the <u>vix level</u>:")
                fo.write("<br/>Total count days (daily POSITIVE change): " + str(positive_vix_count_days))
                fo.write(df_to_html_1_decimal(positive_day_vix_df, include_total_row=False))
                fo.write("<b>HOW TO USE THE TABLE:</b>")
                fo.write("<br/> - row index: range of the VIX")
                fo.write("<br/> - column: daily change [%] (close with respect to the previous day's close)")
                fo.write("<br/> - cell: <u>cumulative probability [%]</u> that the close is <b>lower or equal</b> the change in the column header")

        except Exception as e:
            print('Cannot create the html file:', e)
            sys.exit(1)  # stop the main function with exit code 1

    def print_in_app(self) -> None:
        """
        Print stats in the app.
        :return: None
        """

        st.write(" ")
        st.markdown("<h4 style='text-align: center; '>Price change cumulative probability</h4>", unsafe_allow_html=True)
        st.write(f"Data source: asset={self.__SOURCE}, vix={self.__VIX_SOURCE}")
        st.write(f"The tables in this section contain the **cumulative probability** of the change in price up to a certain level (column).")

        def print_df(df: pd.DataFrame, include_total_row: bool = True) -> None:
            output_df = self.__append_total_count_days_row(df) if include_total_row else df
            st.dataframe(output_df.style.format('{:,.1f}'))

        # Daily and weekly stats
        daily_total_count_days = int(self.__get_total_count_days(self.__daily_change_df))
        st.write("Daily movements (**close** with respect to previous day **close**) "
                 f"- total count days: {daily_total_count_days}")
        print_df(self.__daily_change_df, include_total_row=False)

        friday_to_friday_count_weeks = int(self.__weekly_change_friday_to_friday_count_weeks)
        monday_to_friday_count_weeks = int(self.__weekly_change_monday_to_friday_count_weeks)
        friday_to_friday_days = int(self.__weekly_change_friday_to_friday_count_days)
        monday_to_friday_days = int(self.__weekly_change_monday_to_friday_count_days)
        st.write("Weekly movements (**Friday close** with respect to the previous week **Friday close** or **Monday open**)")
        st.write(f"Friday to Friday: {friday_to_friday_count_weeks} weeks ({friday_to_friday_days} days)")
        st.write(f"Monday to Friday: {monday_to_friday_count_weeks} weeks ({monday_to_friday_days} days)")
        print_df(self.__weekly_change_df)
        st.write("Note: frequency sums to 100% for Monday to Friday, and 100% for Friday to Friday. "
                 "The two are treated separately.")
        monday_positive_count_weeks = int(self.__weekly_if_monday_positive_count_weeks)
        monday_negative_count_weeks = int(self.__weekly_if_monday_negative_count_weeks)
        monday_positive_count_days = int(self.__weekly_if_monday_positive_count_days)
        monday_negative_count_days = int(self.__weekly_if_monday_negative_count_days)
        monday_total_count_weeks = monday_positive_count_weeks + monday_negative_count_weeks
        monday_total_count_days = monday_positive_count_days + monday_negative_count_days
        st.write("Week if Monday positive/negative sample size:")
        st.write(f"Monday positive: {monday_positive_count_weeks} weeks ({monday_positive_count_days} days)")
        st.write(f"Monday negative: {monday_negative_count_weeks} weeks ({monday_negative_count_days} days)")
        st.write(f"Total: {monday_total_count_weeks} weeks ({monday_total_count_days} days)")
        print_df(self.__weekly_change_monday_conditional_df, include_total_row=False)

        if self.__weekly_short_dte_change_df is not None:
            st.write("Short DTE movements: position opened at any day's close and closed at the DTE close.")
            print_df(self.__weekly_short_dte_change_df)

        # Monthly (dte-based) stats
        if self.__monthly_dte_change_df is not None:
            last_open_idx = min(self.__DTE_LONG, self.__number_of_trading_days - 1)
            last_open_date = self.__price_history_df["Date"][last_open_idx].strftime('%d/%m/%Y')
            st.write("Long DTE movements: price change (close to close) in " + str(self.__DTE_LONG) + " trading days. " +
                     str(self.__number_of_trading_days) + " analyzed days. Last: " + last_open_date
                     + ". Open the position at every day close and close at the DTE closing price.")

            print_df(self.__monthly_dte_change_df, include_total_row=False)

        # Gap up / down analysis
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.markdown("<h4 style='text-align: center; '>Gap UP / DOWN analysis </h4>", unsafe_allow_html=True)
        st.write(f"The tables in this section contain the **cumulative probability** of the change in price up to a certain level (column), "
                 f"for a given open gap (market open).")

        st.write(" ")
        st.write("Positive market open (gap up)")
        # check if ND (str) are present when no data are available and change to NAN
        for key in self.__stats_positive_gap.keys():
            col = self.__stats_positive_gap[key]
            for k in col.keys():
                if k != 'gap' and isinstance(col[k], str):
                    self.__stats_positive_gap[key][k] = 0
        gap_up_total_count_days = int(self.__stats_positive_gap[">0 %"]["count days"]) \
            if ">0 %" in self.__stats_positive_gap else 0
        st.write(f"Total count days (positive market opening): {gap_up_total_count_days}")
        gap_up_df = pd.DataFrame([self.__stats_positive_gap[i] for i in self.__stats_positive_gap.keys()])
        gap_up_df.fillna(0)
        gap_up_df = self.__trim_gap_table_columns(gap_up_df)
        print_df(gap_up_df, include_total_row=False)
        st.write("HOW TO USE THE TABLE:")
        st.write("- row index: range of the opening gap-up")
        st.write("- column: daily change [%] (close with respect to the previous day's close)")
        st.write("- cell: cumulative probability [%] that the close is lower or equal the change in the column header")
        st.write(f"- **usage**: observe the open gap. Choose the daily sell put strike based on the close pct with the lowest probability.")

        # check if ND (str) are present when no data are available and change to NAN
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write("Negative market open (gap down)")
        for key in self.__stats_negative_gap.keys():
            col = self.__stats_negative_gap[key]
            for k in col.keys():
                if k != 'gap' and isinstance(col[k], str):
                    self.__stats_negative_gap[key][k] = 0
        gap_down_total_count_days = int(self.__stats_negative_gap[">0 %"]["count days"]) \
            if ">0 %" in self.__stats_negative_gap else 0
        st.write(f"Total count days (negative market opening): {gap_down_total_count_days}")
        gap_down_df = pd.DataFrame([self.__stats_negative_gap[i] for i in self.__stats_negative_gap.keys()])
        gap_down_df.fillna(0)
        gap_down_df = self.__trim_gap_table_columns(gap_down_df)
        print_df(gap_down_df, include_total_row=False)
        st.write("HOW TO USE THE TABLE:")
        st.write("- row index: range of the opening gap-down")
        st.write("- column: daily change [%] (close with respect to the previous day's close")
        st.write("- cell: cumulative probability [%] that the close is lower or equal the change in the column header")
        st.write(f"- **usage**: observe the open gap. Choose the daily sell put strike based on the close pct with the lowest probability.")

        # VIX stats
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.markdown("<h4 style='text-align: center; '>Daily price change given the VIX</h4>", unsafe_allow_html=True)
        st.write(f"The tables in this section contain the **cumulative probability** of the change in price up to a certain level (column), "
                 f"given a certain VIX level.")

        st.write(" ")
        st.write("Daily NEGATIVE change given the vix level:")

        # check if ND (str) are present when no data are available and change to NAN
        for key in self.__dict_daily_change_vix_bins.keys():
            col = self.__dict_daily_change_vix_bins[key]
            for k in col.keys():
                if isinstance(col[k], str):
                    self.__dict_daily_change_vix_bins[key][k] = 0
        negative_day_vix_df = pd.DataFrame([self.__dict_daily_change_vix_bins[i]["cumulative negative"] for i in
                                            self.__dict_daily_change_vix_bins.keys()])
        negative_day_vix_df.set_index("VIX", inplace=True)
        negative_day_vix_df = negative_day_vix_df.fillna(0)
        positive_day_vix_df = pd.DataFrame([self.__dict_daily_change_vix_bins[i]["cumulative positive"] for i in
                                            self.__dict_daily_change_vix_bins.keys()])
        positive_day_vix_df.set_index("VIX", inplace=True)
        positive_day_vix_df = positive_day_vix_df.fillna(0)
        negative_vix_count_days = int(self.__get_total_count_days(negative_day_vix_df))
        positive_vix_count_days = int(self.__get_total_count_days(positive_day_vix_df))
        total_vix_count_days = negative_vix_count_days + positive_vix_count_days
        st.write(f"Total count days across the two VIX tables: {total_vix_count_days}")
        st.write(f"Total count days (daily NEGATIVE change): {negative_vix_count_days}")
        print_df(negative_day_vix_df, include_total_row=False)
        st.write("HOW TO USE THE TABLE:")
        st.write("- row index: range of the VIX")
        st.write("- column: daily change [%] (close with respect to the previous day's close")
        st.write("- cell: **cumulative probability [%]** that the close is **lower or equal** the change in the column header")

        st.write(" ")
        st.write(" ")
        st.write("Daily POSITIVE change according to the vix level:")
        st.write(f"Total count days (daily POSITIVE change): {positive_vix_count_days}")
        print_df(positive_day_vix_df, include_total_row=False)
        st.write("HOW TO USE THE TABLE:")
        st.write("- row index: range of the VIX")
        st.write("- column: daily change [%] (close with respect to the previous day's close)")
        st.write("- cell: **cumulative probability [%]** that the close is **lower or equal** the change in the column header")

    def __make_plot_weekly_change(self) -> go.Figure:
        """
        Make the bar plot of the weekly change of the asset.
        :return: plotly figure
        :rtype: Figure
        """

        week_positive_if_first_positive = {"week": [], "change": []}
        week_negative_if_first_positive = {"week": [], "change": []}
        week_positive_if_first_negative = {"week": [], "change": []}
        week_negative_if_first_negative = {"week": [], "change": []}

        for idx, change in enumerate(self.__weekly_change_first_day_positive):
            if change > 0:
                week_positive_if_first_positive["change"].append(change)
                week_positive_if_first_positive["week"].append(self.__weekly_change_first_day_positive_week_count[idx])
            else:
                week_negative_if_first_positive["change"].append(change)
                week_negative_if_first_positive["week"].append(self.__weekly_change_first_day_positive_week_count[idx])

        for idx, change in enumerate(self.__weekly_change_first_day_negative):
            if change > 0:
                week_positive_if_first_negative["change"].append(change)
                week_positive_if_first_negative["week"].append(self.__weekly_change_first_day_negative_week_count[idx])
            else:
                week_negative_if_first_negative["change"].append(change)
                week_negative_if_first_negative["week"].append(self.__weekly_change_first_day_negative_week_count[idx])

        # Make bar plot
        fig = go.Figure(data=[
            go.Bar(name='positive (day 1:+)',
                   x=week_positive_if_first_positive["week"],
                   y=week_positive_if_first_positive["change"],
                   marker=dict(
                       color='green',
                       line_color='green'
                   ),
                   width=self.__PLOT_COLUMN_WIDTH
                   ),
            go.Bar(name='positive (day 1:-)',
                   x=week_positive_if_first_negative["week"],
                   y=week_positive_if_first_negative["change"],
                   marker=dict(
                       color='green',
                       line_color='red',
                       pattern_shape="/"
                   ),
                   width=self.__PLOT_COLUMN_WIDTH
                   ),
            go.Bar(name='negative (day 1:-)',
                   x=week_negative_if_first_negative["week"],
                   y=week_negative_if_first_negative["change"],
                   marker=dict(
                       color='red',
                       line_color='red'
                   ),
                   width=self.__PLOT_COLUMN_WIDTH
                   ),
            go.Bar(name='negative (day 1:+)',
                   x=week_negative_if_first_positive["week"],
                   y=week_negative_if_first_positive["change"],
                   marker=dict(
                       color='red',
                       line_color='green',
                       pattern_shape="/"
                   ),
                   width=self.__PLOT_COLUMN_WIDTH
                   )
        ])

        weekly_frames = self.__get_weekly_timeframes(min_days=1)
        week_id = list(range(1, len(weekly_frames) + 1))
        week_dates = []
        week_max_vix = []
        for week_df in weekly_frames:
            first_day = week_df["Date"].iloc[0].strftime('%d/%m')
            last_day = week_df["Date"].iloc[-1].strftime('%d/%m - %Y')
            week_dates.append(first_day + "-" + last_day)
            if "VIX" in week_df:
                week_max_vix.append(float(np.max(week_df["VIX"])))
            else:
                week_max_vix.append(0.0)
        week_period_map = {week: period for week, period in zip(week_id, week_dates)}

        def get_periods(week_list: list) -> list:
            return [week_period_map.get(week, "") for week in week_list]

        fig.data[0].customdata = get_periods(week_positive_if_first_positive["week"])
        fig.data[1].customdata = get_periods(week_positive_if_first_negative["week"])
        fig.data[2].customdata = get_periods(week_negative_if_first_negative["week"])
        fig.data[3].customdata = get_periods(week_negative_if_first_positive["week"])
        for trace in fig.data:
            trace.hovertemplate = "Period: %{customdata}<br>Change: %{y:.2f}%<extra></extra>"

        if week_max_vix and any(vix_value > 0 for vix_value in week_max_vix):
            fig.add_trace(
                go.Scatter(
                    name='max VIX',
                    x=week_id,
                    y=week_max_vix,
                    mode='lines',
                    line=dict(color='royalblue', width=2),
                    customdata=week_dates,
                    hovertemplate="Period: %{customdata}<br>Max VIX: %{y:.2f}<extra></extra>",
                    yaxis='y2'
                )
            )

        fig.update_layout(
            title="<b>Weekly change<b>",
            title_x=0.5,
            xaxis=dict(
                tickmode='array',
                tickvals=week_id,
                showticklabels=False
            ),
            yaxis2=dict(
                title="max VIX",
                overlaying='y',
                side='right'
            )
        )

        # Change the bar mode
        fig.update_layout(yaxis=dict(title_text="change [%]"))
        fig.update_xaxes(title_text="week")

        return fig

    def __make_plot_monthly_change(self) -> tuple[go.Figure, list]:
        """
        Make the bar plot of the monthly change of the asset.
        :return: plotly figure
        :rtype: Figure
        :return: list of negative statistics
        :rtype: list
        """

        month_positive = {"day num": [], "change": [], "period": []}
        month_negative = {"day num": [], "change": [], "period": []}

        change_list = list(reversed(self.__change_list_monthly_dte_for_plot_df["change_list"]))
        date_range = list(reversed(self.__change_list_monthly_dte_for_plot_df["date range"]))
        max_vix_list = list(
            reversed(
                self.__change_list_monthly_dte_for_plot_df.get(
                    "max vix",
                    self.__change_list_monthly_dte_for_plot_df.get("avg vix", [])
                )
            )
        )

        for idx, (change, period) in enumerate(zip(change_list, date_range)):
            if change > 0:
                month_positive["change"].append(change)
                month_positive["day num"].append(idx)
                month_positive["period"].append(period)
            else:
                month_negative["change"].append(change)
                month_negative["day num"].append(idx)
                month_negative["period"].append(period)

        # Make bar plot
        fig = go.Figure(data=[
            go.Bar(name='positive change',
                   x=month_positive["day num"],
                   y=month_positive["change"],
                   marker=dict(
                       color='green',
                       line_color='green'
                   ),
                   customdata=month_positive["period"],
                   hovertemplate="Period: %{customdata}<br>Change: %{y:.2f}%<extra></extra>",
                   width=self.__PLOT_COLUMN_WIDTH
                   ),
            go.Bar(name='negative change',
                   x=month_negative["day num"],
                   y=month_negative["change"],
                   marker=dict(
                       color='red',
                       line_color='red'
                   ),
                   customdata=month_negative["period"],
                   hovertemplate="Period: %{customdata}<br>Change: %{y:.2f}%<extra></extra>",
                   width=self.__PLOT_COLUMN_WIDTH
                   )
        ])

        # calculate statistics for negative change
        confidence_interval = self.__mean_confidence_interval(month_negative["change"])

        if len(max_vix_list) == len(change_list) and any(vix_value > 0 for vix_value in max_vix_list):
            fig.add_trace(
                go.Scatter(
                    name='max VIX',
                    x=list(range(len(max_vix_list))),
                    y=max_vix_list,
                    mode='lines',
                    line=dict(color='royalblue', width=2),
                    customdata=date_range,
                    hovertemplate="Period: %{customdata}<br>Max VIX: %{y:.2f}<extra></extra>",
                    yaxis='y2'
                )
            )

        fig.update_layout(
            title="<b>" + str(self.__DTE_LONG) + " DTE change<b>",
            title_x=0.5,
            yaxis2=dict(
                title="max VIX",
                overlaying='y',
                side='right'
            )
        )

        # Change the bar mode
        fig.update_layout(yaxis=dict(title_text="change [%]"))
        fig.update_xaxes(title_text="day")

        return fig, confidence_interval

    @staticmethod
    def __mean_confidence_interval(data: list, confidence: float = 0.95) -> list:
        """
        Calculate the mean and confidence interval of a list of data.
        :param data: list of data
        :type data: list
        :param confidence: confidence level
        :type confidence: float
        :return: list of [mean, standard deviation, lower bound confidence interval, upper bound confidence interval]
        :rtype: list
        """
        n = len(data)
        if n == 0:
            return [0.0, 0.0, 0.0, 0.0]
        if n == 1:
            mean = float(data[0])
            return [mean, 0.0, mean, mean]

        data_array = np.asarray(data, dtype=float)
        mean, standard_deviation = np.mean(data_array), np.std(data_array)

        # Newey-West (HAC) standard error for mean to reduce overconfidence
        # when samples are autocorrelated (e.g., overlapping DTE windows).
        demeaned = data_array - mean
        lag = max(1, int(np.sqrt(n)))
        gamma_0 = float(np.dot(demeaned, demeaned) / n)
        variance_hac = gamma_0
        for lag_i in range(1, lag + 1):
            gamma_i = float(np.dot(demeaned[lag_i:], demeaned[:-lag_i]) / n)
            weight = 1.0 - lag_i / (lag + 1.0)
            variance_hac += 2.0 * weight * gamma_i
        standard_error = np.sqrt(max(variance_hac, 0.0) / n)

        dof = n - 1
        t_crit = np.abs(t.ppf((1 - confidence) / 2., dof))
        return [mean,
                standard_deviation,
                mean - standard_error * t_crit,
                mean + standard_error * t_crit]

    def __build_dte_stats_dataframe(self,
                                    dte: int,
                                    max_change_pct: float,
                                    change_list: list,
                                    drawdown_list: list,
                                    vix_increment_list: list) -> pd.DataFrame:
        """
        Build a DTE stats dataframe from per-trade observations.
        """

        change_positive, change_negative = self.__calc_distribution(change_list, max_change_pct, self.__STEP)

        positive_drawdown_list = [drawdown for change, drawdown in zip(change_list, drawdown_list) if change >= 0]
        negative_drawdown_list = [drawdown for change, drawdown in zip(change_list, drawdown_list) if change < 0]
        positive_vix_increment_list = [
            vix_increment for change, vix_increment in zip(change_list, vix_increment_list) if change >= 0
        ]
        negative_vix_increment_list = [
            vix_increment for change, vix_increment in zip(change_list, vix_increment_list) if change < 0
        ]

        if positive_drawdown_list:
            change_positive["Max drawdown [%]"] = np.min(positive_drawdown_list)
        else:
            change_positive["Max drawdown [%]"] = 0

        if negative_drawdown_list:
            change_negative["Max drawdown [%]"] = np.min(negative_drawdown_list)
        else:
            change_negative["Max drawdown [%]"] = 0

        if positive_vix_increment_list:
            change_positive["Max VIX increment [%]"] = np.max(positive_vix_increment_list)
        else:
            change_positive["Max VIX increment [%]"] = 0

        if negative_vix_increment_list:
            change_negative["Max VIX increment [%]"] = np.max(negative_vix_increment_list)
        else:
            change_negative["Max VIX increment [%]"] = 0

        change_positive["Case"] = str(int(dte)) + "DTE: positive"
        change_negative["Case"] = str(int(dte)) + "DTE: negative"
        # noinspection PyTypeChecker
        change_df = pd.DataFrame.from_dict([change_positive, change_negative])
        change_df.set_index("Case", inplace=True)
        change_df.index.name = None
        return change_df

    def __calc_DTE_statistics(self, dte: int, max_change_pct: float) -> pd.DataFrame:
        """
        Calculate statistics given for any day given a DTE.
        :param dte: date to end (effective trading days)
        :type dte: int
        :param max_change_pct: max weekly change in percentage expressed in the range 0-100
        :type max_change_pct: float
        :return: dataframe with the price change for the selected DTE
        :rtype: pd.DataFrame
        """

        change_list_df, _, _ = self.__calc_change_DTE(dte)
        if dte == self.__DTE_LONG:
            self.__change_list_monthly_dte_for_plot_df = change_list_df
        return self.__build_dte_stats_dataframe(
            dte,
            max_change_pct,
            change_list_df["change_list"],
            change_list_df["drawdown_list"],
            change_list_df["vix_increment_list"],
        )

    def __calc_DTE_statistics_by_vix_regime(self, dte: int, max_change_pct: float) -> dict[str, pd.DataFrame]:
        """
        Calculate DTE statistics conditioned on VIX at position opening.
        """

        change_list_df, _, _ = self.__calc_change_DTE(dte)
        change_list = change_list_df["change_list"]
        drawdown_list = change_list_df["drawdown_list"]
        vix_increment_list = change_list_df["vix_increment_list"]
        open_vix_list = change_list_df["open_vix_list"]

        regimes = {
            "Low volatility (VIX < 20)": lambda vix: vix < 20,
            "Medium volatility (20 <= VIX < 27)": lambda vix: 20 <= vix < 27,
            "High volatility (VIX >= 27)": lambda vix: vix >= 27,
        }

        regime_dfs = {}
        for regime_label, is_in_regime in regimes.items():
            regime_changes = []
            regime_drawdowns = []
            regime_vix_increments = []
            for change, drawdown, vix_increment, open_vix in zip(
                    change_list, drawdown_list, vix_increment_list, open_vix_list):
                if is_in_regime(open_vix):
                    regime_changes.append(change)
                    regime_drawdowns.append(drawdown)
                    regime_vix_increments.append(vix_increment)

            regime_dfs[regime_label] = self.__build_dte_stats_dataframe(
                dte,
                max_change_pct,
                regime_changes,
                regime_drawdowns,
                regime_vix_increments,
            )

        return regime_dfs

    def __calc_day_change_wrt_previous_day(self):
        """
        Calculate open and close change with respect to previous day close.
        """

        open_list = self.__price_history_df["Open"].to_numpy()
        close_list = self.__price_history_df["Close"].to_numpy()
        open_wrt_close_list = [0] * len(open_list)
        close_wrt_close_list = [0] * len(open_list)

        for idx in range(len(open_list)-1):
            previous_day_close = close_list[idx + 1]
            day_open = open_list[idx]
            day_close = close_list[idx]
            close_wrt_close_list[idx] = 100.0 * (day_close - previous_day_close) / previous_day_close
            open_wrt_close_list[idx] = 100.0 * (day_open - previous_day_close) / previous_day_close

        self.__price_history_df["Open wrt close"] = open_wrt_close_list
        self.__price_history_df["Close wrt close"] = close_wrt_close_list

    def __calc_weekly_statistics(self):
        """
        Calculate the statistics of the weekly change in the asset price.
        """

        # Monday to Friday (first to last week days)
        weekly_change_monday_to_friday, weekly_change_monday_to_friday_drawdown_dict, weekly_change_monday_to_friday_vix_change_dict = (
            self.__calc_weekly_movement())
        self.__weekly_change_monday_to_friday_count_weeks = len(weekly_change_monday_to_friday)
        monday_to_friday_positive_dict, monday_to_friday_negative_dict = self.__calc_distribution(
            weekly_change_monday_to_friday,
            self.__WEEK_MAX_CHANGE_PCT,
            self.__STEP)
        monday_to_friday_positive_dict["Case"] = "Monday to Friday: positive"
        monday_to_friday_negative_dict["Case"] = "Monday to Friday: negative"

        # Friday to friday (last to last week days)
        weekly_change_friday_to_friday, weekly_change_friday_to_friday_drawdown_dict, weekly_change_friday_to_friday_vix_change_dict = (
            self.__calc_weekly_friday_to_friday_movement())
        self.__weekly_change_friday_to_friday_count_weeks = len(weekly_change_friday_to_friday)
        friday_to_friday_positive_dict, friday_to_friday_negative_dict = self.__calc_distribution(
            weekly_change_friday_to_friday,
            self.__WEEK_MAX_CHANGE_PCT,
            self.__STEP)
        friday_to_friday_positive_dict["Case"] = "Friday to Friday: positive"
        friday_to_friday_negative_dict["Case"] = "Friday to Friday: negative"

        # Set drawdown and VIX change info. Some buckets can be empty on small/one-sided datasets.
        def set_extra_stats(target_dict: dict, drawdown_list: list, vix_change_list: list):
            if len(drawdown_list) > 0:
                target_dict["Max drawdown [%]"] = np.min(drawdown_list)
                target_dict["Avg drawdown [%]"] = np.mean(drawdown_list)
            else:
                target_dict["Max drawdown [%]"] = 0
                target_dict["Avg drawdown [%]"] = 0

            if len(vix_change_list) > 0:
                target_dict["Max VIX increment [%]"] = np.max(vix_change_list)
                target_dict["Avg VIX increment [%]"] = np.mean(vix_change_list)
            else:
                target_dict["Max VIX increment [%]"] = 0
                target_dict["Avg VIX increment [%]"] = 0

        set_extra_stats(
            monday_to_friday_positive_dict,
            weekly_change_monday_to_friday_drawdown_dict["positive week"],
            weekly_change_monday_to_friday_vix_change_dict["positive week"]
        )
        set_extra_stats(
            monday_to_friday_negative_dict,
            weekly_change_monday_to_friday_drawdown_dict["negative week"],
            weekly_change_monday_to_friday_vix_change_dict["negative week"]
        )
        set_extra_stats(
            friday_to_friday_positive_dict,
            weekly_change_friday_to_friday_drawdown_dict["positive week"],
            weekly_change_friday_to_friday_vix_change_dict["positive week"]
        )
        set_extra_stats(
            friday_to_friday_negative_dict,
            weekly_change_friday_to_friday_drawdown_dict["negative week"],
            weekly_change_friday_to_friday_vix_change_dict["negative week"]
        )

        # noinspection PyTypeChecker
        self.__weekly_change_df = pd.DataFrame.from_dict([monday_to_friday_positive_dict,
                                                          monday_to_friday_negative_dict,
                                                          friday_to_friday_positive_dict,
                                                          friday_to_friday_negative_dict])
        self.__weekly_change_df.set_index("Case", inplace=True)
        self.__weekly_change_df.index.name = None
        self.__weekly_change_df.rename(columns={"count days": "count weeks"}, inplace=True)

    def __calc_weekly_conditional_statistics(self):
        """
        Calculate the statistics of the weekly change of the asset price based on the week's first day price change.
        """

        self.__calc_weekly_if_monday()
        weekly_positive_if_monday_positive_dict, weekly_negative_if_monday_positive_dict = self.__calc_distribution(
            self.__weekly_change_first_day_positive, self.__WEEK_MAX_CHANGE_PCT, self.__STEP)
        weekly_positive_if_monday_negative_dict, weekly_negative_if_monday_negative_dict = self.__calc_distribution(
            self.__weekly_change_first_day_negative, self.__WEEK_MAX_CHANGE_PCT, self.__STEP)

        weekly_positive_if_monday_positive_dict["Case"] = "Week if Monday positive: positive"
        weekly_negative_if_monday_positive_dict["Case"] = "Week if Monday positive: negative"
        weekly_positive_if_monday_negative_dict["Case"] = "Week if Monday negative: positive"
        weekly_negative_if_monday_negative_dict["Case"] = "Week if Monday negative: negative"
        # noinspection PyTypeChecker
        self.__weekly_change_monday_conditional_df = pd.DataFrame.from_dict([weekly_positive_if_monday_positive_dict,
                                                                             weekly_negative_if_monday_positive_dict,
                                                                             weekly_positive_if_monday_negative_dict,
                                                                             weekly_negative_if_monday_negative_dict])
        self.__weekly_change_monday_conditional_df.set_index("Case", inplace=True)
        self.__weekly_change_monday_conditional_df.index.name = None
        self.__weekly_change_monday_conditional_df.rename(columns={"count days": "count weeks"}, inplace=True)

    def __calc_number_of_weeks_in_year(self, year: int) -> list:
        """
        Get the first weeknumber and the last weeknumber for the selected year.
        :param year: year for the calculation of the number of weeks
        :type year: int
        :return: list with first and last weeknumber in the input year
        :rtype list
        """

        return [int(np.min(self.__price_history_df.loc[self.__price_history_df["Year"] == year]["Week number"])),
                int(np.max(self.__price_history_df.loc[self.__price_history_df["Year"] == year]["Week number"]))]

    def __get_weekly_timeframes(self, min_days: int = 1) -> list[pd.DataFrame]:
        """
        Return weekly slices sorted in chronological order.
        Grouping is done by (Year, Week number), then each week is sorted by date ascending.
        :param min_days: minimum number of rows required to include a week
        :type min_days: int
        :return: list of weekly dataframes
        :rtype: list
        """

        if self.__price_history_df is None or self.__price_history_df.empty:
            return []

        weekly_frames = []
        grouped = self.__price_history_df.groupby(["Year", "Week number"], sort=False)
        for _, week_df in grouped:
            week_df_sorted = week_df.sort_values("Date", ascending=True)
            if len(week_df_sorted) >= min_days:
                weekly_frames.append(week_df_sorted)

        weekly_frames.sort(key=lambda week_df: week_df["Date"].iloc[0])
        return weekly_frames

    def __calc_weekly_if_monday(self):
        """
        Calculate the weekly change depending on Monday's (or first weekday) change.
        """

        self.__weekly_change_first_day_positive = []
        self.__weekly_change_first_day_negative = []
        self.__weekly_change_first_day_positive_week_count = []
        self.__weekly_change_first_day_negative_week_count = []
        self.__weekly_if_monday_positive_count_weeks = 0
        self.__weekly_if_monday_negative_count_weeks = 0
        self.__weekly_if_monday_positive_count_days = 0
        self.__weekly_if_monday_negative_count_days = 0

        week_counter = 0
        for week_df in self.__get_weekly_timeframes(min_days=1):
            week_counter += 1
            week_open = week_df["Open"].iloc[0]
            week_close = week_df["Close"].iloc[-1]
            first_day_change = 0
            week_change = 0
            if week_open != 0:
                week_change = 100.0 * (week_close - week_open) / week_open
                first_day_change = 100.0 * (week_df["Close"].iloc[0] - week_open) / week_open
            if first_day_change > 0:
                self.__weekly_change_first_day_positive.append(week_change)
                self.__weekly_change_first_day_positive_week_count.append(week_counter)
                self.__weekly_if_monday_positive_count_weeks += 1
                self.__weekly_if_monday_positive_count_days += len(week_df)
            else:
                self.__weekly_change_first_day_negative.append(week_change)
                self.__weekly_change_first_day_negative_week_count.append(week_counter)
                self.__weekly_if_monday_negative_count_weeks += 1
                self.__weekly_if_monday_negative_count_days += len(week_df)

    @staticmethod
    def __calc_drawdown(asset_data_selected_timeframe_df: pd.DataFrame, price_open: float) -> float:
        """
        Calculate the drawdown in the asset price.
        :param asset_data_selected_timeframe_df
        :type: pd.DataFrame
        :param price_open
        :type: float
        :return: drawdown
        :rtype: float
        """
        price_minimum = asset_data_selected_timeframe_df["Low"].values.min()
        return 100. * (price_minimum - price_open) / price_open

    @staticmethod
    def __calc_max_change_vix(asset_data_selected_timeframe_df: pd.DataFrame, vix_start: float) -> float:
        """
        Calculate the max change in VIX.
        :param asset_data_selected_timeframe_df
        :type: pd.DataFrame
        :param vix_start
        :type: float
        :return: vix change
        :rtype: float
        """
        if vix_start == 0:
            return 0
        vix_max = asset_data_selected_timeframe_df["VIX"].values.max()
        return 100. * (vix_max - vix_start) / vix_start

    def __calc_weekly_movement(self) -> tuple:
        """
        Calculate the weekly movement of the ticker. Weekdays shall be at least 2.
        :return: list with the weekly changes and its drawdown
        :rtype: tuple
        """

        change_monday_to_friday_list = []
        analyzed_days = 0
        drawdown_dict = {"positive week": [],
                         "negative week": []}
        change_vix_dict = {"positive week": [],
                           "negative week": []}
        for week_df in self.__get_weekly_timeframes(min_days=2):
            analyzed_days += len(week_df)
            week_open = week_df["Open"].iloc[0]
            week_close = week_df["Close"].iloc[-1]
            vix_open = week_df["VIX"].iloc[0] if "VIX" in week_df else 0
            if week_close >= week_open:
                drawdown_dict["positive week"].append(self.__calc_drawdown(week_df, week_open))
                change_vix_dict["positive week"].append(self.__calc_max_change_vix(week_df, vix_open))
            else:
                drawdown_dict["negative week"].append(self.__calc_drawdown(week_df, week_open))
                change_vix_dict["negative week"].append(self.__calc_max_change_vix(week_df, vix_open))
            change = 0
            if week_open != 0:
                change = 100.0 * (week_close - week_open) / week_open
            change_monday_to_friday_list.append(change)
        self.__weekly_change_monday_to_friday_count_days = analyzed_days
        return change_monday_to_friday_list, drawdown_dict, change_vix_dict

    def __calc_weekly_friday_to_friday_movement(self) -> tuple:
        """
        Calculate the weekly movement of the ticker (last day to last day).
        The current week shall have at least 4 weekdays.
        :return: list with the weekly changes and drawdown
        :rtype: tuple
        """

        change_friday_to_friday_list = []
        analyzed_days = 0
        drawdown_dict = {"positive week": [],
                         "negative week": []}
        change_vix_dict = {"positive week": [],
                           "negative week": []}
        weekly_frames = self.__get_weekly_timeframes(min_days=1)
        for idx in range(1, len(weekly_frames)):
            week_df = weekly_frames[idx]
            # current week shall have at least 4 weekdays
            if len(week_df) < 4:
                continue
            analyzed_days += len(week_df)
            previous_week_df = weekly_frames[idx - 1]
            week_close = week_df["Close"].iloc[-1]
            previous_week_close = previous_week_df["Close"].iloc[-1]
            vix_open = previous_week_df["VIX"].iloc[-1] if "VIX" in previous_week_df else 0
            if week_close >= previous_week_close:
                drawdown_dict["positive week"].append(self.__calc_drawdown(week_df, previous_week_close))
                change_vix_dict["positive week"].append(self.__calc_max_change_vix(week_df, vix_open))
            else:
                drawdown_dict["negative week"].append(self.__calc_drawdown(week_df, previous_week_close))
                change_vix_dict["negative week"].append(self.__calc_max_change_vix(week_df, vix_open))
            # calculate weekly changes
            change = 0
            if previous_week_close != 0:
                change = 100.0 * (week_close - previous_week_close) / previous_week_close
            change_friday_to_friday_list.append(change)
        self.__weekly_change_friday_to_friday_count_days = analyzed_days
        return change_friday_to_friday_list, drawdown_dict, change_vix_dict

    def __calc_change_DTE(self, dte: int) -> tuple:
        """
        Calculate the price change (close to close) given an input DTE (Date To End) of every day of the price data.
        :param dte: Date To End (effective trading days until option expiration)
        :type dte: int
        :return: dataframe with the DTE changes for each day of the selected period
        :rtype: tuple
        """

        drawdown_dict = {"positive week": [],
                         "negative week": []}
        change_vix_dict = {"positive week": [],
                           "negative week": []}

        change_list = [0] * (self.__number_of_trading_days - dte)
        date_range = [0] * (self.__number_of_trading_days - dte)
        max_vix_list = [0] * (self.__number_of_trading_days - dte)
        drawdown_list = [0] * (self.__number_of_trading_days - dte)
        vix_increment_list = [0] * (self.__number_of_trading_days - dte)
        open_vix_list = [0] * (self.__number_of_trading_days - dte)
        # element 0 is the top of the dataframe (most recent date)
        # change is calculated as: (CLOSE(DTE)-CLOSE(today)) / CLOSE(today)
        # to_list to speed-up the loop over the dataframe
        price_close_list = self.__price_history_df["Close"].to_list()
        date_list = self.__price_history_df["Date"].to_list()
        vix_list = self.__price_history_df["VIX"].to_list()
        daily_low_list = self.__price_history_df["Low"].to_list()
        # loop in inverse order on a descending-date dataframe:
        # open is at idx (older), close-after-dte is at idx-dte (newer)
        for idx in range(self.__number_of_trading_days - 1, dte - 1, -1):
            open_price_at_close = price_close_list[idx]
            close_after_dte_days = price_close_list[idx - dte]
            vix_within_dte = vix_list[(idx - dte):idx+1]
            low_within_dte = daily_low_list[(idx - dte):idx+1]
            vix_open = vix_list[idx]
            max_vix = np.max(vix_within_dte)
            vix_increase_max = 0
            if vix_open != 0:
                vix_increase_max = 100 * (max_vix - vix_open) / vix_open
            lowest_low = np.min(low_within_dte)
            change = 0
            drawdown = 0
            if open_price_at_close != 0:
                change = 100.0 * (close_after_dte_days - open_price_at_close) / open_price_at_close
                drawdown = 100.0 * (lowest_low - open_price_at_close) / open_price_at_close
                if change >= 0:
                    drawdown_dict["positive week"].append(drawdown)
                    change_vix_dict["positive week"].append(vix_increase_max)
                else:
                    drawdown_dict["negative week"].append(drawdown)
                    change_vix_dict["negative week"].append(vix_increase_max)
            change_list[idx - dte] = change
            drawdown_list[idx - dte] = drawdown
            vix_increment_list[idx - dte] = float(vix_increase_max)
            open_vix_list[idx - dte] = float(vix_open)
            date_range[idx - dte] = date_list[idx].strftime('%d/%m - ') + date_list[idx - dte].strftime('%d/%m/%Y')
            max_vix_list[idx - dte] = float(max_vix)
        return {
            "change_list": change_list,
            "date range": date_range,
            "max vix": max_vix_list,
            "drawdown_list": drawdown_list,
            "vix_increment_list": vix_increment_list,
            "open_vix_list": open_vix_list,
        }, drawdown_dict, change_vix_dict

    @staticmethod
    def __calc_positive_negative_change_lists(change_list: list) -> tuple:
        """
        Split the input change list into positive change list and negative change list.
        :param change_list: list of price changes
        :type change_list: list
        :returns: lists of positive and negative changes
        :rtype: tuple
        """

        positive_change_list = sorted(list(filter(lambda change: change >= 0.0, change_list)))
        negative_change_list = sorted(list(filter(lambda change: change < 0.0, change_list)))
        return positive_change_list, negative_change_list

    def __split_for_day_performance(self):
        """
        Split database according to the daily open and close with respect to the previous close.
        """
        # for idx in range(1, len(self.__price_history_df)):
        # Split for positive and negative day (close vs. close previous day)
        self.__day_positive_close = \
            self.__price_history_df.loc[self.__price_history_df["Close wrt close"] > 0]["Close wrt close"].values
        self.__day_negative_close = \
            self.__price_history_df.loc[self.__price_history_df["Close wrt close"] < 0]["Close wrt close"].values

        # Split for gap-up and gap-down at opening (vs. close previous day)
        self.__day_gapup_df = \
            self.__price_history_df.loc[self.__price_history_df["Open wrt close"] > 0]
        self.__day_gapdown_df = \
            self.__price_history_df.loc[self.__price_history_df["Open wrt close"] < 0]

    def __calc_distribution(self, change_list: list, pct_max: float, step: float) -> tuple:
        """
        Calculate the distributions of positive and negative changes.
        Calculate the cumulative probability distribution.
        :param change_list: list of price changes
        :type change_list: list
        :param pct_max: max price change percentage in the range 0-100
        :type pct_max: float
        :param: step: step of percentage to calculate the cumulative distribution
        :type: float
        :returns: dictionaries of positive and negative changes with the cumulative distribution up to pct_max
        :rtype: tuple
        """

        positive_change, negative_change = self.__calc_positive_negative_change_lists(change_list)
        num_positive = len(positive_change)
        num_negative = len(negative_change)
        pct_list = np.linspace(0, pct_max, int((pct_max / step)) + 1)
        dict_positive = {"frequency [%]": 0, "count days": num_positive}
        dict_negative = {"frequency [%]": 0, "count days": num_negative}
        if num_negative + num_positive > 0:
            dict_positive["frequency [%]"] = 100.0 * num_positive / (num_positive + num_negative)
            dict_negative["frequency [%]"] = 100.0 * num_negative / (num_positive + num_negative)
        cumulative_positive = 0
        cumulative_negative = 0

        # calculate cumulative distribution
        for pct_low, pct_high in zip(pct_list, pct_list[1:]):
            if num_positive > 0:
                pct_positive = 100.0 * len(
                    list(filter(lambda x: pct_low <= x < pct_high, positive_change))) / num_positive
            else:
                pct_positive = 0
            cumulative_positive += pct_positive
            dict_positive[str(int(pct_high)) + "% change"] = cumulative_positive
            if num_negative > 0:
                pct_negative = 100.0 * len(
                    list(filter(lambda x: pct_low <= abs(x) < pct_high, negative_change))) / num_negative
            else:
                pct_negative = 0
            cumulative_negative += pct_negative
            dict_negative[str(int(pct_high)) + "% change"] = cumulative_negative

        if num_positive > 0:
            dict_positive["max change [%]"] = np.max(positive_change)
            dict_positive["avg change [%]"] = np.mean(positive_change)
        else:
            dict_positive["max change [%]"] = 0
            dict_positive["avg change [%]"] = 0

        if num_negative > 0:
            dict_negative["max change [%]"] = np.min(negative_change)
            dict_negative["avg change [%]"] = np.mean(negative_change)
        else:
            dict_negative["max change [%]"] = 0
            dict_negative["avg change [%]"] = 0

        return dict_positive, dict_negative

    def get_price_history(self) -> pd.DataFrame:
        """
        Return price history dataframe.
        :return: dataframe of the price history of the asset
        :rtype: pd.DataFrame
        """

        return self.__price_history_df

    def get_source(self) -> str:
        """
        Return source for price query.
        :return: price history query source
        :rtype: str
        """
        return self.__SOURCE

    def get_vix_source(self) -> str:
        """
        Return source for VIX query.
        :return: VIX history query source
        :rtype: str
        """
        return self.__VIX_SOURCE
