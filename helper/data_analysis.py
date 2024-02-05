import yfinance as yf
import math
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import plotly.graph_objects as go
import sys

import datetime
from scipy.stats import t


class PriceAnalysis:
    """
    Class to analyze historical pricing data of a specific asset.
    """

    def __init__(self, ticker: str,
                 start: datetime,
                 end: datetime,
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
        :param path_to_report: path where the report is saved
        :type path_to_report: str
        :param stats_vix: query vix data or not
        :type stats_vix: bool
        :param do_plot: plot graphs in the html file
        :type do_plot: false
        """

        self.__SOURCE = 'stooq'

        self.__DO_PLOT = do_plot
        self.__STATS_VIX = stats_vix

        self.__WEEK_TRADING_DAYS = 5
        self.__MONTH_TRADING_DAYS = 23
        self.__WEEK_MAX_CHANGE_PCT = 6
        self.__MONTH_MAX_CHANGE_PCT = 12
        self.__NUMBER_WEEKS_PER_YEAR = 52
        self.__DTE_LONG = 23
        self.__STEP = 1  # step to calculate the cumulative distribution
        self.__BINS_DAILY_CHANGE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        self.__BINS_VIX = [vix for vix in range(10,41,3)]

        self.__PLOT_COLUMN_WIDTH = 0.75

        self.__STEP_GAP_OPEN = 0.25
        self.__BIN_CLOSE_PCT = 0.5
        self.__MAX_GAP = 2.5

        self.__NO__DATA_INDICATOR = "ND"

        self.__ticker = ticker
        self.__date_start = start
        self.__date_end = end
        self.__number_of_days = (end - start).days
        self.__number_of_weeks = math.ceil(self.__number_of_days / 7)
        self.__number_of_trading_days = 0

        self.__PATH_TO_HTML = path_to_report
        if start.year == end.year:
            self.__years_analysis = str(int(start.year))
        else:
            self.__years_analysis = str(int(start.year)) + '-' + str(int(end.year))
        self.__filename = self.__PATH_TO_HTML + self.__ticker \
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
        self.__weekly_change_df = None
        self.__weekly_change_monday_conditional_df = None
        self.__weekly_dte_change_df = None
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
        self.query_vix()

        # Step 2: calculate daily statistics
        self.__calc_daily_statistics()
        self.__calc_daily_statistics_vix()

        # Step 3: calculate weekly statistics
        self.__calc_weekly_statistics()
        self.__calc_weekly_conditional_statistics()
        if self.__number_of_days >= self.__WEEK_TRADING_DAYS:
            self.__weekly_dte_change_df = self.__calc_DTE_statistics(self.__WEEK_TRADING_DAYS,
                                                                     self.__WEEK_MAX_CHANGE_PCT)

        # Step 4: calculate monthly statistics
        if self.__number_of_days >= self.__DTE_LONG:
            self.__monthly_dte_change_df = self.__calc_DTE_statistics(self.__DTE_LONG,
                                                                      self.__MONTH_MAX_CHANGE_PCT)

        # Step 5: calculate gap-ups and -downs statistics
        self.__calc_stats_gapup_down()

        # Step 5: make html report
        self.__write_html()
        print("Report written in: " + self.__filename)

    def data_sanity_check(self) -> list:
        """
        Check if any NaN in the queried data.
        :return: list of dates where NaN were found
        :rtype: list
        """
        COLUMNS_TO_CHECK = ["Date", "Week number", "Year", "Open", "Close"]
        for key in COLUMNS_TO_CHECK:
            if self.__price_history_df[key].isnull().values.any():
                idx_nan = np.argwhere(np.isnan(self.__price_history_df[key].values))
                return self.__price_history_df[key].iloc(idx_nan).values
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

        cumulative_prob_daily_positive_dict["Day"] = "positive"
        cumulative_prob_daily_negative_dict["Day"] = "negative"

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
        min_close_positive_open = np.min(close_positive_open)
        max_close_positive_open = np.max(close_positive_open)
        min_close_negative_open = np.min(close_negative_open)
        max_close_negative_open = np.max(close_negative_open)

        x_cpf_positive_open = []
        x = int(min_close_positive_open*self.__BIN_CLOSE_PCT)/self.__BIN_CLOSE_PCT+self.__BIN_CLOSE_PCT
        while x < max_close_positive_open:
            x_cpf_positive_open.append(x)
            x += self.__BIN_CLOSE_PCT
        if len(x_cpf_positive_open) > 0:
            if x_cpf_positive_open[-1] < max_close_positive_open:
                x_cpf_positive_open.append(max_close_positive_open)
            elif x_cpf_positive_open[-1] > max_close_positive_open:
                x_cpf_positive_open = max_close_positive_open

        x_cpf_negative_open = []
        x = int(max_close_negative_open*self.__BIN_CLOSE_PCT)/self.__BIN_CLOSE_PCT - self.__BIN_CLOSE_PCT
        while x > min_close_negative_open:
            x_cpf_negative_open.append(x)
            x -= self.__BIN_CLOSE_PCT
        if len(x_cpf_negative_open) > 0:
            if x_cpf_negative_open[-1] > min_close_negative_open:
                x_cpf_negative_open.append(min_close_negative_open)
            elif x_cpf_negative_open[-1] > min_close_negative_open:
                x_cpf_negative_open = min_close_negative_open

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
                    gap-self.__STEP_GAP_OPEN)))]
            key = "]"+str(gap-self.__STEP_GAP_OPEN)+"; "+str(gap)+"]%"
            self.__stats_positive_gap[str(gap) + " %"] = {"gap": key}
            cpf = self.__calc_cpf(close_list, x_cpf_positive_open)
            if len(cpf) > 0:
                for i, close in enumerate(x_cpf_positive_open):
                    self.__stats_positive_gap[str(gap) + " %"][str(int(close*10)/10)+"%"] = cpf[i]
            else:
                self.__stats_positive_gap[str(gap) + " %"][str(int(0 * 10) / 10) + "%"] = 0

        # above the max gap considered in the list
        close_list = [daily_close_pct[i] for i in range(len(daily_close_pct)) if daily_open_pct[i] > gap_positive_list[-1]]
        key = ">" + str(gap_positive_list[-1]) + " %"
        self.__stats_positive_gap[">+" + str(gap_positive_list[-1]) + " %"] = {"gap": key}
        cpf = self.__calc_cpf(close_list, x_cpf_positive_open)
        if len(cpf) > 0:
            for i, close in enumerate(x_cpf_positive_open):
                self.__stats_positive_gap[">+" + str(gap_positive_list[-1]) + " %"][str(int(close * 10) / 10) + "%"] = cpf[i]
        else:
            self.__stats_positive_gap[">+" + str(gap_positive_list[-1]) + " %"][str(int(0 * 10) / 10) + "%"] = 0

        # all positive gap-ups
        close_list = [daily_close_pct[i] for i in range(len(daily_close_pct)) if daily_open_pct[i] > 0]
        key = "positive open %"
        self.__stats_positive_gap[">0 %"] = {"gap": key}
        cpf = self.__calc_cpf(close_list, x_cpf_positive_open)
        for i, close in enumerate(x_cpf_positive_open):
            self.__stats_positive_gap[">0 %"][str(int(close * 10) / 10) + "%"] = cpf[i]

            # gap up
            for gap in gap_positive_list:
                close_list = [daily_close_pct[i] for i in range(len(daily_close_pct)) if ((daily_open_pct[i] <= gap) and (daily_open_pct[i] > (
                        gap - self.__STEP_GAP_OPEN)))]
                key = "]" + str(gap - self.__STEP_GAP_OPEN) + "; " + str(gap) + "]%"
                self.__stats_positive_gap[str(gap) + " %"] = {"gap": key}
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
            self.__stats_positive_gap[">+" + str(gap) + " %"] = {"gap": key}
            if close_list:
                cpf = self.__calc_cpf(close_list, x_cpf_positive_open)
            else:
                cpf = [self.__NO__DATA_INDICATOR] * len(x_cpf_positive_open)
            for idx, close_pct in enumerate(x_cpf_positive_open):
                self.__stats_positive_gap[">+" + str(gap) + " %"][str(int(close_pct * 10) / 10) + "%"] = cpf[idx]

            # all positive gap-ups
            close_list = [daily_close_pct[i] for i in range(len(daily_close_pct)) if daily_open_pct[i] > 0]
            key = "positive open %"
            self.__stats_positive_gap[">0 %"] = {"gap": key}
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
            self.__stats_negative_gap[str(gap) + " %"] = {"gap": key}
            if close_list:
                cpf = self.__calc_cpf([-i for i in close_list], [-i for i in x_cpf_negative_open])
            else:
                cpf = [self.__NO__DATA_INDICATOR]*len(x_cpf_negative_open)
            for idx, close_pct in enumerate(x_cpf_negative_open):
                self.__stats_negative_gap[str(gap) + " %"][str(int(close_pct * 10) / 10) + "%"] = cpf[idx]

        # above the max gap considered in the list
        close_list = [daily_close_pct[i] for i in range(len(daily_close_pct)) if daily_open_pct[i] < gap_positive_list[-1]]
        gap = gap_negative_list[-1]
        key = ">" + str(gap) + " %"
        self.__stats_negative_gap[">+" + str(gap) + " %"] = {"gap": key}
        if close_list:
            cpf = self.__calc_cpf([-i for i in close_list], [-i for i in x_cpf_negative_open])
        else:
            cpf = [self.__NO__DATA_INDICATOR] * len(x_cpf_negative_open)
        for idx, close_pct in enumerate(x_cpf_negative_open):
            self.__stats_negative_gap[">+" + str(gap) + " %"][str(int(close_pct * 10) / 10) + "%"] = cpf[idx]

        # all negative gap-ups
        close_list = [daily_close_pct[i] for i in range(len(daily_close_pct)) if daily_open_pct[i] < 0]
        key = "negative open %"
        self.__stats_negative_gap[">0 %"] = {"gap": key}
        if close_list:
            cpf = self.__calc_cpf([-i for i in close_list], [-i for i in x_cpf_negative_open])
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

        dict_cumulative_dist = {"frequency [%]": 0}
        num_days = len(data)
        if data[0] >= 0:  # positive data if min >=0, otherwise negative data
            for pct in self.__BINS_DAILY_CHANGE:
                dict_cumulative_dist[str(int(pct*10)/10) + "% change"] = \
                    100.0 * len([j for j in data if j <= pct]) / num_days
        else:
            for pct in self.__BINS_DAILY_CHANGE:
                dict_cumulative_dist[str(int(pct*10)/10) + "% change"] = \
                    100.0 * len([j for j in data if j >= -pct]) / num_days

        return dict_cumulative_dist

    def update_analysis_period(self, start: datetime, end: datetime):
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
        self.__filename = self.__PATH_TO_HTML + self.__ticker + '_' \
                                              + self.__years_analysis \
                                              + '_update' + self.__date_end.strftime('%d%m%Y') \
                                              + '.html'

    def query_asset_price(self):
        """
        Query data of the ticker for the input timerange from the source database.
        """

        try:
            try:
                self.__price_history_df = yf.Ticker(self.__ticker)
                self.__price_history_df = self.__price_history_df.history(start=self.__date_start,
                                                                          end=self.__date_end)
            except Exception:
                self.__price_history_df = web.DataReader(self.__ticker,
                                                         self.__SOURCE,
                                                         self.__date_start,
                                                         self.__date_end, 10)
        except Exception as e:
            print('Cannot query historical data:', e)
            sys.exit(1)  # stop the main function with exit code 1

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
        for index, row in self.__price_history_df.iterrows():
            d = pd.to_datetime(index)
            weekday_list.append(d.weekday())
            weeknumber_list.append(d.isocalendar()[1])
            years_list.append(d.year)
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

        try:
            # vix_history_df = yf.download('^VIX', start = self.__date_start_vix, end=self.__date_end)
            vix_history_df = yf.Ticker("^VIX")
            vix_history_df = vix_history_df.history(start=self.__date_start, end=self.__date_end)

        except Exception as e:
            print('Cannot query historical data of VIX:', e)
            sys.exit(1)  # stop the main function with exit code 1

        vix_history_df["Date"] = [d.date() for d in vix_history_df.index.to_list()]
        vix_history_df.set_index("Date", inplace=True)
        # Append VIX column after the last column of the dataframe
        self.__price_history_df["VIX"] = 0.
        vix_date_list = vix_history_df.index.values
        for vix_date in vix_date_list:
            vix = vix_history_df.loc[vix_date]["Close"]
            self.__price_history_df.loc[self.__price_history_df['Date'] == vix_date, "VIX"] = vix

        # Fill unavailable vix data with the value of the previous day. Loop from oldest (idx=len(list)) to the most recent day (idx=0)
        vix_history_close_list = self.__price_history_df["VIX"].to_list()
        for idx in range(len(vix_history_close_list)-1, -1, -1):
            if vix_history_close_list[idx] == 0:
                vix_history_close_list[idx] = vix_history_close_list[idx+1]
        self.__price_history_df["VIX"] = vix_history_close_list


    def __calc_daily_statistics_vix(self):
        """
        Calculate the cumulative probability of daily change (close to close) according to the VIX levels.
        Data in each bin interval are selected such that vix>vix_min & vix<=vix_max.
        :return: None
        """

        # Check values of the vix <= min(BINS_VIX)
        filtered_df = self.__price_history_df.loc[self.__price_history_df['VIX'] <= min(self.__BINS_VIX)]
        vix_key = str(min(self.__BINS_VIX))
        if not filtered_df.empty:
            positive_daily_change = filtered_df.loc[filtered_df["Close wrt close"] > 0]["Close wrt close"].tolist()
            negative_daily_change = filtered_df.loc[filtered_df["Close wrt close"] < 0]["Close wrt close"].tolist()
            count_positive_days = len(positive_daily_change)
            count_negative_days = len(negative_daily_change)
            count_days = count_positive_days + count_negative_days
            if len(negative_daily_change) > 0:
                self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"] = self.__calc_cumulative_probability(negative_daily_change)
                self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"]["frequency [%]"] = 100. * count_negative_days / count_days
                self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"]["count days"] = count_negative_days
            if len(positive_daily_change) > 0:
                self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"] = self.__calc_cumulative_probability(positive_daily_change)
                self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"]["frequency [%]"] = 100. * count_positive_days / count_days
                self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"]["count days"] = count_positive_days
        self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"]["VIX"] = vix_key+"]"
        self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"]["VIX"] = vix_key+"]"

        # Check values of the vix > max(BINS_VIX)
        filtered_df = self.__price_history_df.loc[self.__price_history_df['VIX'] > max(self.__BINS_VIX)]
        vix_key = str(max(self.__BINS_VIX)) + "+"
        if not filtered_df.empty:
            positive_daily_change = filtered_df.loc[filtered_df["Close wrt close"] > 0]["Close wrt close"].tolist()
            negative_daily_change = filtered_df.loc[filtered_df["Close wrt close"] < 0]["Close wrt close"].tolist()
            count_positive_days = len(positive_daily_change)
            count_negative_days = len(negative_daily_change)
            count_days = count_positive_days + count_negative_days
            if len(negative_daily_change) > 0:
                self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"] = self.__calc_cumulative_probability(negative_daily_change)
                self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"]["frequency [%]"] = 100. * count_negative_days / count_days
                self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"]["count days"] = count_negative_days
            if len(positive_daily_change) > 0:
                self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"] = self.__calc_cumulative_probability(positive_daily_change)
                self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"]["frequency [%]"] = 100. * count_positive_days / count_days
                self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"]["count days"] = count_positive_days
        self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"]["VIX"] = "]" + vix_key
        self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"]["VIX"] = "]" + vix_key

        # Check the vix intervals
        for idx in range(len(self.__BINS_VIX) - 1):
            vix_min = self.__BINS_VIX[idx]
            vix_max = self.__BINS_VIX[idx + 1]
            filtered_df = self.__price_history_df.loc[(self.__price_history_df['VIX'] > vix_min) & (self.__price_history_df['VIX'] <= vix_max)]
            vix_key = str(vix_max)
            if not filtered_df.empty:
                positive_daily_change = filtered_df.loc[filtered_df["Close wrt close"] > 0]["Close wrt close"].tolist()
                negative_daily_change = filtered_df.loc[filtered_df["Close wrt close"] < 0]["Close wrt close"].tolist()
                count_positive_days = len(positive_daily_change)
                count_negative_days = len(negative_daily_change)
                count_days = count_positive_days + count_negative_days
                if len(negative_daily_change) > 0:
                    self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"] = self.__calc_cumulative_probability(negative_daily_change)
                    self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"]["frequency [%]"] = 100. * count_negative_days / count_days
                    self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"]["count days"] = count_negative_days
                if len(positive_daily_change) > 0:
                    self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"] = self.__calc_cumulative_probability(positive_daily_change)
                    self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"]["frequency [%]"] = 100. * count_positive_days / count_days
                    self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"]["count days"] = count_positive_days
            self.__dict_daily_change_vix_bins[vix_key]["cumulative negative"]["VIX"] = "]"+str(vix_min)+"; " + str(vix_max) + "]"
            self.__dict_daily_change_vix_bins[vix_key]["cumulative positive"]["VIX"] = "]"+str(vix_min)+"; " + str(vix_max) + "]"


    def __write_html(self):
        """
        Write output file with all the statistics.
        """

        try:
            with open(os.path.expanduser(self.__filename), 'w') as fo:
                fo.write("<html>\n<head>\n<title> \nOutput Data in an HTML file \
                          </title>\n</head> <body><h1><center>" + self.__ticker + "</center></h1>\n</body></html>")
                fo.write("Statistical analysis " + self.__ticker + " " + self.__years_analysis)
                fo.write("<br/>Time period: " + self.__date_start.strftime('%d/%m/%Y'))
                fo.write(" to " + self.__date_end.strftime('%d/%m/%Y'))
                fo.write(" (" + str(int(self.__number_of_weeks)) + " weeks)")
                fo.write("<br/>Number of trading days analyzed: " + str(self.__number_of_trading_days))
                fo.write("<br/>Documented created on: " + datetime.datetime.today().strftime('%d/%m/%Y'))
                fo.write('<br/>' + "Tables contain the <u>cumulative probability</u> of change.")
                # ================================= Daily and Weekly STATS ===================================
                fo.write('<br/><br/>')
                fo.write("<center><b>Daily and weekly change stats</b></center>")
                fo.write('<br/>' + '<br/>' + "Daily change (CLOSE with respect to the previous day's CLOSE)")
                fo.write('<br/>')
                fo.write(self.__daily_change_df.to_html().replace('<td>', '<td align="center">'))
                fo.write('<br/>' + '<br/>' + "Weekly change (Friday CLOSE with respect to the previous week Friday's CLOSE or Monday's OPEN)")
                fo.write('<br/>')
                fo.write(self.__weekly_change_df.to_html().replace('<td>', '<td align="center">'))
                fo.write('<br/>')
                fo.write(self.__weekly_change_monday_conditional_df.to_html().replace('<td>', '<td align="center">'))
                fo.write('<br/>')
                # ================================= DTE STATS ===================================
                fo.write('<br/><br/>')
                fo.write("<center><b>DTE analysis for sell put debit</b></center>")
                fo.write('<br/><br/>')
                if self.__weekly_dte_change_df is not None:
                    fo.write(self.__weekly_dte_change_df.to_html().replace('<td>', '<td align="center">'))

                # monthly dte
                if self.__monthly_dte_change_df is not None:
                    fo.write('<br/>' + '<br/>' + "Change in " + str(self.__MONTH_TRADING_DAYS))
                    fo.write(" DTE (effective trading days, Daily OPEN)")
                    fo.write("<br>" + str(self.__number_of_trading_days) + " analyzed days (last OPEN ")
                    fo.write(self.__price_history_df["Date"][self.__number_of_trading_days - self.__DTE_LONG].strftime(
                        '%d/%m/%Y') + ")")
                    fo.write('<br/>')
                    fo.write(self.__monthly_dte_change_df.to_html().replace('<td>', '<td align="center">'))
                    figure_dte_change, negative_change_stats = self.__make_plot_monthly_change()
                    fo.write('Stats ' + str(self.__MONTH_TRADING_DAYS) + ' DTE negative change:')
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
                    if self.__number_of_days > self.__MONTH_TRADING_DAYS:
                        fo.write('<br/>')
                        fo.write((figure_dte_change.to_html(full_html=False, include_plotlyjs='cdn')))

                # ================================= GAP UP / DOWN STATS ===================================
                fo.write('<br/><br/><br/>')
                fo.write("<center><b>Open gap-up / down analysis</b></center>")
                fo.write('<br/>')
                fo.write("<br/><b>Cumulative probability of the daily change</b> when a <u>positive market opening</u> occurs:")
                fo.write('<br/><br/>')
                gap_up_df = pd.DataFrame([self.__stats_positive_gap[i] for i in self.__stats_positive_gap.keys()])
                gap_up_df.set_index("gap", inplace=True)
                gap_up_df.index.name = None
                fo.write(gap_up_df.to_html().replace('<td>', '<td align="center">'))
                fo.write("<b>HOW TO USE THE TABLE:</b>")
                fo.write("<br/> - row index: range of the opening gap-up")
                fo.write("<br/> - column: daily change [%] (close with respect to the previous day's close")
                fo.write("<br/> - cell: <u>cumulative probability [%]</u> that the close is <b>lower or equal</b> the change in the column header")
                fo.write("<br> - USAGE: observe the open gap. Choose the daily sell put strike based on the close pct with the lowest probability.")

                fo.write('<br/><br/>')
                fo.write("<br/><b>Cumulative probability of the daily change</b> when a <u>negative market opening</u> occurs:")
                fo.write('<br/><br/>')
                gap_down_df = pd.DataFrame([self.__stats_negative_gap[i] for i in self.__stats_negative_gap.keys()])
                gap_down_df.set_index("gap", inplace=True)
                gap_down_df.index.name = None
                fo.write(gap_down_df.to_html().replace('<td>', '<td align="center">'))
                fo.write("<b>HOW TO USE THE TABLE:</b>")
                fo.write("<br/> - row index: range of the opening gap-down")
                fo.write("<br/> - column: daily change [%] (close with respect to the previous day's close")
                fo.write("<br/> - cell: <u>cumulative probability [%]</u> that the close is <b>lower or equal</b> the change in the column header")
                fo.write("<br> - USAGE: observe the open gap. Choose the daily sell put strike based on the close pct with the lowest probability.")

                # ================================= DAILY CHANGE VS. VIX STATS ===================================
                fo.write('<br/><br/><br/>')
                fo.write("<center><b>Daily change according to VIX</b></center>")
                fo.write('<br/>')
                fo.write("<br/><b>Cumulative probability</b> of the <b>daily NEGATIVE change</b> according to the <u>vix level</u>:")
                negative_day_vix_df = pd.DataFrame([self.__dict_daily_change_vix_bins[i]["cumulative negative"] for i in
                                            self.__dict_daily_change_vix_bins.keys()])
                negative_day_vix_df.set_index("VIX", inplace=True)
                negative_day_vix_df.index.name = None
                negative_day_vix_df = negative_day_vix_df.fillna(0)
                fo.write(negative_day_vix_df.to_html().replace('<td>', '<td align="center">'))
                fo.write("<b>HOW TO USE THE TABLE:</b>")
                fo.write("<br/> - row index: range of the VIX")
                fo.write("<br/> - column: daily change [%] (close with respect to the previous day's close")
                fo.write("<br/> - cell: <u>cumulative probability [%]</u> that the close is <b>lower or equal</b> the change in the column header")
                fo.write('<br/>')
                fo.write("<br/><b>Cumulative probability</b> of the <b>daily POSITIVE change</b> according to the <u>vix level</u>:")
                positive_day_vix_df = pd.DataFrame([self.__dict_daily_change_vix_bins[i]["cumulative positive"] for i in
                                                    self.__dict_daily_change_vix_bins.keys()])
                positive_day_vix_df.set_index("VIX", inplace=True)
                positive_day_vix_df.index.name = None
                positive_day_vix_df = positive_day_vix_df.fillna(0)
                fo.write(positive_day_vix_df.to_html().replace('<td>', '<td align="center">'))
                fo.write("<b>HOW TO USE THE TABLE:</b>")
                fo.write("<br/> - row index: range of the VIX")
                fo.write("<br/> - column: daily change [%] (close with respect to the previous day's close")
                fo.write("<br/> - cell: <u>cumulative probability [%]</u> that the close is <b>lower or equal</b> the change in the column header")


        except Exception as e:
            print('Cannot create the html file:', e)
            sys.exit(1)  # stop the main function with exit code 1

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

        week_dates = []
        week_id = list(range(1, self.__number_of_weeks+1))
        for week_number in week_id:
            week_df = self.__price_history_df.loc[self.__price_history_df["Week number"] == week_number]
            first_day = week_df["Date"].iloc[0].strftime('%d/%m')
            last_day = week_df["Date"].iloc[-1].strftime('%d/%m - %Y')
            week_dates.append(first_day + "-" + last_day)

        fig.update_layout(
            title="<b>Weekly change<b>",
            title_x=0.5,
            xaxis=dict(
                tickmode='array',
                tickvals=week_id,
                ticktext=week_dates
            )
        )

        # Change the bar mode
        fig.update_yaxes(title_text="change [%]")
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

        month_positive = {"day num": [], "change": []}
        month_negative = {"day num": [], "change": []}

        for idx, change in enumerate(self.__change_list_monthly_dte_for_plot_df["change_list"]):
            if change > 0:
                month_positive["change"].append(change)
                month_positive["day num"].append(idx)
            else:
                month_negative["change"].append(change)
                month_negative["day num"].append(idx)

        # Make bar plot
        fig = go.Figure(data=[
            go.Bar(name='positive change',
                   x=month_positive["day num"],
                   y=month_positive["change"],
                   marker=dict(
                       color='green',
                       line_color='green'
                   ),
                   width=self.__PLOT_COLUMN_WIDTH
                   ),
            go.Bar(name='negative change',
                   x=month_negative["day num"],
                   y=month_negative["change"],
                   marker=dict(
                       color='red',
                       line_color='red'
                   ),
                   width=self.__PLOT_COLUMN_WIDTH
                   )
        ])

        # calculate statistics for negative change
        confidence_interval = self.__mean_confidence_interval(month_negative["change"])
        fig.update_layout(
            title="<b>" + str(self.__MONTH_TRADING_DAYS) + " DTE change<b>",
            title_x=0.5,
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, len(self.__change_list_monthly_dte_for_plot_df["date range"]))),
                ticktext=self.__change_list_monthly_dte_for_plot_df["date range"]
            )
        )

        # Change the bar mode
        fig.update_yaxes(title_text="change [%]")
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
        dof = n - 1
        mean, standard_deviation = np.mean(data), np.std(data)
        t_crit = np.abs(t.ppf((1 - confidence) / 2., dof))
        return [mean,
                standard_deviation,
                mean - standard_deviation*t_crit/np.sqrt(n),
                mean + standard_deviation*t_crit/np.sqrt(n)]

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

        change_list_df, drawdown_dict, vix_change_dict = self.__calc_change_DTE(dte)
        if dte == self.__MONTH_TRADING_DAYS:
            self.__change_list_monthly_dte_for_plot_df = change_list_df
        change_list = change_list_df["change_list"]
        change_positive, change_negative = self.__calc_distribution(change_list, max_change_pct, self.__STEP)
        change_positive["Max drawdown [%]"] = np.min(drawdown_dict["positive week"])
        change_negative["Max drawdown [%]"] = np.min(drawdown_dict["negative week"])
        change_positive["Max VIX increment [%]"] = np.max(vix_change_dict["positive week"])
        change_negative["Max VIX increment [%]"] = np.max(vix_change_dict["negative week"])
        change_positive["Case"] = "Daily to " + str(int(dte)) + "DTE: positive"
        change_negative["Case"] = "Daily to " + str(int(dte)) + "DTE: negative"
        # noinspection PyTypeChecker
        change_df = pd.DataFrame.from_dict([change_positive, change_negative])
        change_df.set_index("Case", inplace=True)
        change_df.index.name = None
        return change_df

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
        monday_to_friday_positive_dict, monday_to_friday_negative_dict = self.__calc_distribution(
            weekly_change_monday_to_friday,
            self.__WEEK_MAX_CHANGE_PCT,
            self.__STEP)
        monday_to_friday_positive_dict["Case"] = "Monday to Friday: positive"
        monday_to_friday_negative_dict["Case"] = "Monday to Friday: negative"

        # Friday to friday (last to last week days)
        weekly_change_friday_to_friday, weekly_change_friday_to_friday_drawdown_dict, weekly_change_friday_to_friday_vix_change_dict = (
            self.__calc_weekly_friday_to_friday_movement())
        friday_to_friday_positive_dict, friday_to_friday_negative_dict = self.__calc_distribution(
            weekly_change_friday_to_friday,
            self.__WEEK_MAX_CHANGE_PCT,
            self.__STEP)
        friday_to_friday_positive_dict["Case"] = "Friday to Friday: positive"
        friday_to_friday_negative_dict["Case"] = "Friday to Friday: negative"

        # Set drawdown and vix change info
        monday_to_friday_positive_dict["Max drawdown [%]"] = np.min(weekly_change_monday_to_friday_drawdown_dict["positive week"])
        monday_to_friday_positive_dict["Avg drawdown [%]"] = np.mean(weekly_change_monday_to_friday_drawdown_dict["positive week"])
        monday_to_friday_positive_dict["Max VIX increment [%]"] = np.max(weekly_change_monday_to_friday_vix_change_dict["positive week"])
        monday_to_friday_positive_dict["Avg VIX increment [%]"] = np.mean(weekly_change_monday_to_friday_vix_change_dict["positive week"])
        monday_to_friday_negative_dict["Max drawdown [%]"] = np.min(weekly_change_monday_to_friday_drawdown_dict["negative week"])
        monday_to_friday_negative_dict["Avg drawdown [%]"] = np.mean(weekly_change_monday_to_friday_drawdown_dict["negative week"])
        monday_to_friday_negative_dict["Max VIX increment [%]"] = np.max(weekly_change_monday_to_friday_vix_change_dict["negative week"])
        monday_to_friday_negative_dict["Avg VIX increment [%]"] = np.mean(weekly_change_monday_to_friday_vix_change_dict["negative week"])
        friday_to_friday_positive_dict["Max drawdown [%]"] = np.min(weekly_change_friday_to_friday_drawdown_dict["positive week"])
        friday_to_friday_positive_dict["Avg drawdown [%]"] = np.mean(weekly_change_friday_to_friday_drawdown_dict["positive week"])
        friday_to_friday_positive_dict["Max VIX increment [%]"] = np.max(weekly_change_friday_to_friday_vix_change_dict["positive week"])
        friday_to_friday_positive_dict["Avg VIX increment [%]"] = np.mean(weekly_change_friday_to_friday_vix_change_dict["positive week"])
        friday_to_friday_negative_dict["Max drawdown [%]"] = np.min(weekly_change_friday_to_friday_drawdown_dict["negative week"])
        friday_to_friday_negative_dict["Avg drawdown [%]"] = np.mean(weekly_change_friday_to_friday_drawdown_dict["negative week"])
        friday_to_friday_negative_dict["Max VIX increment [%]"] = np.max(weekly_change_friday_to_friday_vix_change_dict["negative week"])
        friday_to_friday_negative_dict["Avg VIX increment [%]"] = np.mean(weekly_change_friday_to_friday_vix_change_dict["negative week"])

        # noinspection PyTypeChecker
        self.__weekly_change_df = pd.DataFrame.from_dict([monday_to_friday_positive_dict,
                                                          monday_to_friday_negative_dict,
                                                          friday_to_friday_positive_dict,
                                                          friday_to_friday_negative_dict])
        self.__weekly_change_df.set_index("Case", inplace=True)
        self.__weekly_change_df.index.name = None

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

    def __calc_weekly_if_monday(self):
        """
        Calculate the weekly change depending on Monday's (or first weekday) change.
        """

        week_counter = 0
        for year in self.__years_list:
            weeks_in_the_year = self.__calc_number_of_weeks_in_year(year)
            for week_number in range(weeks_in_the_year[0], weeks_in_the_year[1] + 1):
                week_counter += 1
                # filter the selected week and the previous week
                week_df = self.__price_history_df.loc[self.__price_history_df["Week number"] == week_number]
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
                else:
                    self.__weekly_change_first_day_negative.append(week_change)
                    self.__weekly_change_first_day_negative_week_count.append(week_counter)



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
        vix_max = asset_data_selected_timeframe_df["VIX"].values.max()
        return 100. * (vix_max - vix_start) / vix_start


    def __calc_weekly_movement(self) -> tuple:
        """
        Calculate the weekly movement of the ticker. Weekdays shall be at least 2.
        :return: list with the weekly changes and its drawdown
        :rtype: tuple
        """

        change_monday_to_friday_list = []
        drawdown_dict = {"positive week": [],
                         "negative week": []}
        change_vix_dict = {"positive week": [],
                         "negative week": []}
        for year in self.__years_list:
            week_range_in_the_year = self.__calc_number_of_weeks_in_year(year)
            for week_number in range(week_range_in_the_year[0], week_range_in_the_year[1] + 1):
                # filter the selected week and the previous week
                week_df = self.__price_history_df.loc[self.__price_history_df["Week number"] == week_number]
                week_open = week_df["Open"].iloc[0]
                week_close = week_df["Close"].iloc[-1]
                vix_open = week_df["VIX"].iloc[0]
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
        return change_monday_to_friday_list, drawdown_dict, change_vix_dict

    def __calc_weekly_friday_to_friday_movement(self) -> tuple:
        """
        Calculate the weekly movement of the ticker (last day to last day).
        The current week shall have at least 4 weekdays.
        :return: list with the weekly changes and drawdown
        :rtype: tuple
        """

        change_friday_to_friday_list = []
        drawdown_dict = {"positive week": [],
                         "negative week": []}
        change_vix_dict = {"positive week": [],
                         "negative week": []}
        for year in self.__years_list:
            weeks_in_the_year = self.__calc_number_of_weeks_in_year(year)
            for week_number in range(weeks_in_the_year[0] + 1, weeks_in_the_year[1] + 1):
                # filter the selected week and the previous week
                week_df = self.__price_history_df.loc[self.__price_history_df["Week number"] == week_number]
                previous_week_df = self.__price_history_df.loc[
                    self.__price_history_df["Week number"] == week_number - 1]
                week_close = week_df["Close"].iloc[-1]
                previous_week_close = previous_week_df["Close"].iloc[-1]
                vix_open = previous_week_df["VIX"].iloc[-1]
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
        # element 0 is the top of the dataframe (most recent date)
        # change is calculated as: (CLOSE(DTE)-CLOSE(today)) / CLOSE(today)
        # to_list to speed-up the loop over the dataframe
        price_close_list = self.__price_history_df["Close"].to_list()
        date_list = self.__price_history_df["Date"].to_list()
        vix_list = self.__price_history_df["VIX"].to_list()
        daily_low_list = self.__price_history_df["Low"].to_list()
        # loop in inverse order: the open price is at location idx - dte, closing price at location idx
        for idx in range(self.__number_of_trading_days - 1, dte - 1, -1):
            close_after_dte_days = price_close_list[idx]
            open_price_at_close = price_close_list[idx - dte]
            vix_within_dte = vix_list[(idx - dte):idx+1]
            low_within_dte = daily_low_list[(idx - dte):idx+1]
            vix_open = vix_within_dte[0]
            max_vix = np.max(vix_within_dte)
            vix_increase_max = 0
            if vix_open != 0:
                vix_increase_max = 100 * (max_vix - vix_open) / vix_open
            lowest_low = np.min(low_within_dte)
            change = 0
            drawdown = 0
            if open_price_at_close != 0:
                change = 100.0 * (close_after_dte_days - open_price_at_close) / open_price_at_close
                if lowest_low != 0:
                    drawdown = 100 * (lowest_low - open_price_at_close) / lowest_low
                if change >= 0:
                    drawdown_dict["positive week"].append(drawdown)
                    change_vix_dict["positive week"].append(vix_increase_max)
                else:
                    drawdown_dict["negative week"].append(drawdown)
                    change_vix_dict["negative week"].append(vix_increase_max)
            change_list[idx - dte] = change
            date_range[idx - dte] = date_list[idx].strftime('%d/%m - ') + date_list[idx - dte].strftime('%d/%m/%Y')
        return {"change_list": change_list, "date range": date_range}, drawdown_dict, change_vix_dict

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
        dict_positive = {"frequency [%]": 0}
        dict_negative = {"frequency [%]": 0}
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
