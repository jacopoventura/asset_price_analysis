# Copyright (c) 2024 Jacopo Ventura

import datetime

import numpy as np
import os
import pandas as pd
# import pandas_datareader.data as web
import plotly.graph_objects as go
import sys
# import plotly.express as px
# from scipy.stats import t


class EsPriceAnalysis:
    """
    Class to analyze historical pricing data of ES.
    """

    def __init__(self, folder: str, es_data_filename: str):
        """
        Initialize the class PriceAnalysis with ticker, start and end time for the price analysis.
        """
        self.__KEY_FIRST_REBOUND = "Rebound 1  08:30-11"  # Rebound at PP/S/R +/- 2pt
        self.__KEY_SECOND_REBOUND = "Rebound 2 08:30-11"  # Rebound at PP/S/R +/- 2pt
        self.__KEY_PT_DEEP_REBOUND = "Pt deep rebound"
        self.__KEY_NO_PERFECT_REBOUND = "No perfect rebound"  # Rebound after more than 2pt after cross or no rebound
        self.__KEY_TREND = "Trend since 8:30"
        self.__KEY_ES_PP = "ES & PP"

        self.__ES_DATA_FILENAME = es_data_filename
        self.__FOLDER = folder

        self.__PATH_FINAL_REPORT = self.__FOLDER + '/ES_stats.html'

        self.__FACTOR_ROUND = 10
        self.__BIN_RANGE_CPF = 5
        self.__BIN_TREND_CPF = 10

        self.__x_cpf_time = [
            datetime.time(hour=9, minute=30),
            datetime.time(hour=10, minute=00),
            datetime.time(hour=10, minute=30),
            datetime.time(hour=11, minute=00),
            datetime.time(hour=11, minute=30),
            datetime.time(hour=12, minute=00),
            datetime.time(hour=12, minute=30),
            datetime.time(hour=13, minute=00),
            datetime.time(hour=13, minute=30),
            datetime.time(hour=14, minute=00),
            datetime.time(hour=14, minute=30),
            datetime.time(hour=15, minute=00)
        ]

        # dictionary for trend change analysis
        self.__stats_reversal = {"General": {},
                                 "reversal day": {},
                                 "uptrend day": {},
                                 "downtrend day": {},
                                 "range day": {}
                                 }

        self.__stats_hour_reversal = {
            "General": {},
            "Uptrend": {},
            "Downtrend": {},
            "Range": {}
            }

        # dictionary to store the statistics on the DeMark's pivot points
        self.__stats_pivot_points = {"General": {},
                                     "Days with demark": {},
                                     "Support": {},
                                     "Resistance": {},
                                     "Pivot": {}}

        self.__stats_deep_rebound = {"General": {},
                                     "Support": {},
                                     "Resistance": {},
                                     "Pivot": {}}

        self.__stats_no_rebound = {"Range": {},
                                   "ES PP": {},
                                   "Point": {}}

        self.__range_then_reversal = {" ": 0,
                                      "% uptrend reversal": 0,
                                      "% downtrend reversal": 0,
                                      "avg. pt range no reversal": 0,
                                      "avg. pt range then uptrend": 0,
                                      "avg. pt range then downtrend": 0
                                      }

        self.__stats_range_range = {"Range": {},
                                    "Range ES<PP": {},
                                    "Range ES>PP": {}}

        self.__stats_range_uptrend = {"Uptrend": {},
                                      "Uptrend ES<PP": {},
                                      "Uptrend ES>PP": {}}

        self.__stats_range_downtrend = {"Downtrend": {},
                                        "Downtrend ES<PP": {},
                                        "Downtrend ES>PP": {}}

        self.__time_stats_no_rebound = {"Support": {},
                                        "Resistance": {},
                                        "Pivot": {}}

        self.__body_stats_no_rebound = {"Support": {},
                                        "Resistance": {},
                                        "Pivot": {}}

        self.__es_price_df = []
        self.__num_days = 0
        self.__es_uptrend_df = []
        self.__num_days_uptrend = 0
        self.__es_downtrend_df = []
        self.__num_days_downtrend = 0
        self.__es_range_df = []
        self.__num_days_range = 0
        self.__es_uptrend_then_reversal_df = []
        self.__es_downtrend_then_reversal_df = []
        self.__es_range_then_reversal_df = []

        self.__pct_uptrend_then_reversal = 0
        self.__pct_downtrend_then_reversal = 0
        self.__pct_range_then_reversal = 0
        self.__es_price_with_reversal_df = []

        self.__es_higher_pp_start = []
        self.__num_days_es_higher_pp_start = 0
        self.__es_lower_pp_start = []
        self.__num_days_es_lower_pp_start = 0

        self.__es_higher_pp_start_then_reversal_df = []
        self.__es_lower_pp_start_then_reversal_df = []

        self.__pct_es_higher_pp_start_then_reversal = 0
        self.__pct_es_lower_pp_start_then_reversal = 0

    def run(self):
        """
        Run the whole analysis and save the final document with all the statistics.
        """

        # Step 1: get historical price data for the selected time period and check data quality
        self.query_data()

        self.__analyze_reversal()

        self.__analyze_pivot_points()

        self.__analyze_range()

        self.__analyze_no_rebound()

        # Step 4: make html report
        self.__write_html()
        print("Report written in: " + self.__PATH_FINAL_REPORT)

    def query_data(self):
        """
        Query data of the ticker for the input timerange from the source database.
        """

        try:
            self.__es_price_df = pd.read_excel(self.__FOLDER + self.__ES_DATA_FILENAME, sheet_name='ES movement')
            self.__num_days = len(self.__es_price_df.index)
        except Exception as e:
            print('Cannot query historical data:', e)
            sys.exit(1)  # stop the main function with exit code 1

    def __count_point(self, point: int, list_perfect_rebound: list, list_deep_rebound: list, list_no_rebound: list) -> tuple[list, list, list]:
        """
        Count occurrences of the point.
        """
        return list_perfect_rebound.count(point), list_deep_rebound.count(point), list_no_rebound.count(point)

    def __round(self, n: int):
        return int(n * self.__FACTOR_ROUND) / self.__FACTOR_ROUND

    def __analyze_pivot_points(self):
        """
        Analysis of ES reaction at pivot points (DeMark).
        The database contains two columns first and second rebounds: here ES bounces at the point +/- 2pt. These points
        are collected as perfect bounces.
        The column "no rebound" contains the points at which (+/- 2pt) ES did not make the rebound.
        In some of these cases, ES bounced later within 15pt from the point (therefore not causing a stop hedge with
        MasteringSP500). If the column point bounce contains a number, the bounce is deep and within 15pt.
        Otherwise, no rebound happened.
        """

        # Get row indexes of days with PP interaction
        rows_first_rebound = self.__es_price_df[self.__es_price_df[self.__KEY_FIRST_REBOUND].notnull()].index.tolist()
        rows_second_rebound = self.__es_price_df[self.__es_price_df[self.__KEY_SECOND_REBOUND].notnull()].index.tolist()
        rows_perfect_rebound = rows_first_rebound + rows_second_rebound
        rows_deep_rebound = self.__es_price_df[self.__es_price_df[self.__KEY_PT_DEEP_REBOUND].notnull()].index.tolist()
        rows_no_perfect_rebound = self.__es_price_df[self.__es_price_df[self.__KEY_NO_PERFECT_REBOUND].notnull()].index.tolist()
        rows_no_rebound = [i for i in rows_no_perfect_rebound if i not in rows_deep_rebound]

        # Get number of Demark points
        self.__stats_pivot_points["General"]["count perfect rebound"] = len(rows_perfect_rebound)
        self.__stats_pivot_points["General"]["count deep rebound"] = len(rows_deep_rebound)
        self.__stats_pivot_points["General"]["count rebound"] = self.__stats_pivot_points["General"]["count perfect rebound"] + self.__stats_pivot_points["General"]["count deep rebound"]
        self.__stats_pivot_points["General"]["count no perfect rebound"] = len(rows_no_perfect_rebound)
        self.__stats_pivot_points["General"]["count no rebound"] = len(rows_no_rebound)
        self.__stats_pivot_points["General"]["count points"] = self.__stats_pivot_points["General"]["count rebound"] + self.__stats_pivot_points["General"]["count no rebound"]
        self.__stats_pivot_points["General"]["total days"] = len(self.__es_price_df["Date"])

        point_perfect_rebound = [self.__es_price_df.iloc[i][self.__KEY_FIRST_REBOUND] for i in rows_first_rebound] + [self.__es_price_df.iloc[i][self.__KEY_SECOND_REBOUND] for i in rows_second_rebound]
        point_deep_rebound = [self.__es_price_df.iloc[i][self.__KEY_NO_PERFECT_REBOUND] for i in rows_deep_rebound]
        pt_deep_rebound = [self.__es_price_df.iloc[i][self.__KEY_PT_DEEP_REBOUND] for i in rows_deep_rebound]
        pt_deep_rebound_pivot = [self.__es_price_df.iloc[i][self.__KEY_PT_DEEP_REBOUND] for i in rows_deep_rebound if self.__es_price_df.iloc[i][
            self.__KEY_NO_PERFECT_REBOUND] == "PP"]
        pt_deep_rebound_support = [self.__es_price_df.iloc[i][self.__KEY_PT_DEEP_REBOUND] for i in rows_deep_rebound if self.__es_price_df.iloc[i][
            self.__KEY_NO_PERFECT_REBOUND] == "S"]
        pt_deep_rebound_resistance = [self.__es_price_df.iloc[i][self.__KEY_PT_DEEP_REBOUND] for i in rows_deep_rebound if self.__es_price_df.iloc[
            i][self.__KEY_NO_PERFECT_REBOUND] == "R"]
        point_no_rebound = [self.__es_price_df.iloc[i][self.__KEY_NO_PERFECT_REBOUND] for i in rows_no_rebound]
        days_with_perfect_rebound = rows_first_rebound
        days_with_deep_rebound = rows_deep_rebound
        days_with_no_rebound = rows_no_rebound
        days_with_no_perfect_rebound = rows_no_perfect_rebound
        days_with_rebound = list(set(days_with_perfect_rebound + days_with_deep_rebound))
        days_with_demark = list(set(days_with_rebound + days_with_no_rebound))
        days_with_no_demark = [i for i in range(self.__stats_pivot_points["General"]["total days"]) if i not in days_with_demark]

        demark_days_with_rebound = days_with_rebound
        demark_days_with_rebound_only = [i for i in days_with_rebound if i not in days_with_no_rebound]
        demark_days_with_perfect_rebound_only = [i for i in days_with_perfect_rebound if i not in days_with_no_perfect_rebound]
        demark_days_with_deep_rebound_only = [i for i in days_with_deep_rebound if i not in days_with_perfect_rebound]
        demark_days_with_no_rebound_only = [i for i in days_with_no_rebound if i not in days_with_rebound]
        demark_days_with_no_rebound = days_with_no_rebound

        # Analyze data
        self.__stats_pivot_points["Days with demark"]["count"] = len(days_with_demark)
        self.__stats_pivot_points["Days with demark"]["with rebound"] = 100. * len(demark_days_with_rebound) / self.__stats_pivot_points["Days with demark"]["count"]
        self.__stats_pivot_points["Days with demark"]["only rebound"] = 100. * len(demark_days_with_rebound_only) / self.__stats_pivot_points["Days with demark"]["count"]
        self.__stats_pivot_points["Days with demark"]["only perfect rebound"] = 100. * len(demark_days_with_perfect_rebound_only) / self.__stats_pivot_points["Days with demark"]["count"]
        self.__stats_pivot_points["Days with demark"]["only deep rebound"] = 100. * len(demark_days_with_deep_rebound_only) / self.__stats_pivot_points["Days with demark"]["count"]
        self.__stats_pivot_points["Days with demark"]["only no rebound"] = 100. * len(demark_days_with_no_rebound_only) / self.__stats_pivot_points["Days with demark"]["count"]
        self.__stats_pivot_points["Days with demark"]["with no rebound"] = 100. * len(demark_days_with_no_rebound) / self.__stats_pivot_points["Days with demark"]["count"]
        self.__stats_pivot_points["Days with demark"]["with perfect rebound"] = 100. * len(days_with_perfect_rebound) / \
                                                                           self.__stats_pivot_points[
                                                                               "Days with demark"]["count"]
        self.__stats_pivot_points["Days with demark"]["with deep rebound"] = 100. * len(days_with_deep_rebound) / \
                                                                                self.__stats_pivot_points[
                                                                                    "Days with demark"]["count"]

        self.__stats_pivot_points["General"]["pct days with rebound"] = 100. * len(days_with_rebound) / self.__stats_pivot_points["General"]["total days"]
        self.__stats_pivot_points["General"]["pct days no demark"] = 100. * len(days_with_no_demark) / \
                                                                        self.__stats_pivot_points["General"][
                                                                            "total days"]
        self.__stats_pivot_points["General"]["pct days with demark"] = 100. * len(days_with_demark) / \
                                                                     self.__stats_pivot_points["General"][
                                                                         "total days"]

        self.__stats_pivot_points["General"]["pct rebound"] = 100. * (self.__stats_pivot_points["General"]["count rebound"]) / self.__stats_pivot_points["General"]["count points"]
        self.__stats_pivot_points["General"]["pct no rebound"] = 100. * self.__stats_pivot_points["General"]["count no rebound"] / self.__stats_pivot_points["General"]["count points"]
        self.__stats_pivot_points["General"]["pct perfect rebound"] = 100. * self.__stats_pivot_points["General"]["count perfect rebound"] / self.__stats_pivot_points["General"]["count points"]
        self.__stats_pivot_points["General"]["pct deep rebound"] = 100. * self.__stats_pivot_points["General"]["count deep rebound"] / self.__stats_pivot_points["General"]["count points"]
        self.__stats_pivot_points["General"]["pct perfect rebound if rebound"] = 100. * self.__stats_pivot_points["General"]["count perfect rebound"] / self.__stats_pivot_points["General"]["count rebound"]
        self.__stats_pivot_points["General"]["pt deep rebound"] = [self.__es_price_df.iloc[i][self.__KEY_PT_DEEP_REBOUND] for i in rows_deep_rebound]
        count_pp_perfect_rebound, count_pp_deep_rebound, count_pp_no_rebound = self.__count_point("PP",
                                                                                                  point_perfect_rebound,
                                                                                                  point_deep_rebound,
                                                                                                  point_no_rebound)
        count_pp_total = count_pp_perfect_rebound + count_pp_deep_rebound + count_pp_no_rebound
        self.__stats_pivot_points["Pivot"]["pct pivot perfect rebound"] = 100. * count_pp_perfect_rebound / self.__stats_pivot_points["General"]["count perfect rebound"]
        self.__stats_pivot_points["Pivot"]["pct pivot deep rebound"] = 100. * count_pp_deep_rebound / self.__stats_pivot_points["General"]["count deep rebound"]
        self.__stats_pivot_points["Pivot"]["pct pivot rebound"] = 100. * (count_pp_perfect_rebound + count_pp_deep_rebound) / self.__stats_pivot_points["General"]["count rebound"]
        self.__stats_pivot_points["Pivot"]["pct pivot no rebound"] = 100. * count_pp_no_rebound / self.__stats_pivot_points["General"]["count no rebound"]
        self.__stats_pivot_points["Pivot"]["pct point"] = 100. * count_pp_total / self.__stats_pivot_points["General"]["count points"]
        self.__stats_pivot_points["Pivot"]["pct perfect rebound"] = 100. * count_pp_perfect_rebound / count_pp_total
        self.__stats_pivot_points["Pivot"]["pct deep rebound"] = 100. * count_pp_deep_rebound / count_pp_total
        self.__stats_pivot_points["Pivot"]["pct rebound"] = 100. * (count_pp_perfect_rebound + count_pp_deep_rebound) / count_pp_total
        self.__stats_pivot_points["Pivot"]["pct no rebound"] = 100. * count_pp_no_rebound / count_pp_total
        self.__stats_pivot_points["Pivot"]["pt deep rebound"] = [self.__es_price_df.iloc[i][self.__KEY_PT_DEEP_REBOUND] for i in rows_deep_rebound
                                                                 if self.__es_price_df.iloc[i][self.__KEY_NO_PERFECT_REBOUND] == "PP"]

        count_support_perfect_rebound, count_support_deep_rebound, count_support_no_rebound = self.__count_point("S",
                                                                                                                 point_perfect_rebound,
                                                                                                                 point_deep_rebound,
                                                                                                                 point_no_rebound)
        count_support_total = count_support_perfect_rebound + count_support_deep_rebound + count_support_no_rebound
        self.__stats_pivot_points["Support"]["pct support perfect rebound"] = 100. * count_support_perfect_rebound / self.__stats_pivot_points["General"]["count perfect rebound"]
        self.__stats_pivot_points["Support"]["pct support deep rebound"] = 100. * count_support_deep_rebound / self.__stats_pivot_points["General"]["count deep rebound"]
        self.__stats_pivot_points["Support"]["pct support rebound"] = 100. * (count_support_perfect_rebound + count_support_deep_rebound) / self.__stats_pivot_points["General"]["count rebound"]
        self.__stats_pivot_points["Support"]["pct support no rebound"] = 100. * count_support_no_rebound / self.__stats_pivot_points["General"]["count no rebound"]
        self.__stats_pivot_points["Support"]["pct point"] = 100. * count_support_total / self.__stats_pivot_points["General"]["count points"]
        self.__stats_pivot_points["Support"]["pct perfect rebound"] = 100. * count_support_perfect_rebound / count_support_total
        self.__stats_pivot_points["Support"]["pct deep rebound"] = 100. * count_support_deep_rebound / count_support_total
        self.__stats_pivot_points["Support"]["pct rebound"] = 100. * (
                    count_support_perfect_rebound + count_support_deep_rebound) / count_support_total
        self.__stats_pivot_points["Support"]["pct no rebound"] = 100. * count_support_no_rebound / count_support_total
        self.__stats_pivot_points["Support"]["pt deep rebound"] = [self.__es_price_df.iloc[i][self.__KEY_PT_DEEP_REBOUND]
                                                                   for i in rows_deep_rebound if self.__es_price_df.iloc[i][
                                                                       self.__KEY_NO_PERFECT_REBOUND] == "S"]

        count_resistance_perfect_rebound, count_resistance_deep_rebound, count_resistance_no_rebound = self.__count_point("R",
                                                                                                  point_perfect_rebound,
                                                                                                  point_deep_rebound,
                                                                                                  point_no_rebound)
        count_resistance_total = count_resistance_perfect_rebound + count_resistance_deep_rebound + count_resistance_no_rebound
        self.__stats_pivot_points["Resistance"]["pct resistance perfect rebound"] = 100. * count_resistance_perfect_rebound / self.__stats_pivot_points["General"]["count perfect rebound"]
        self.__stats_pivot_points["Resistance"]["pct resistance deep rebound"] = 100. * count_resistance_deep_rebound / self.__stats_pivot_points["General"]["count deep rebound"]
        self.__stats_pivot_points["Resistance"]["pct resistance rebound"] = 100. * (count_resistance_perfect_rebound + count_resistance_deep_rebound) / self.__stats_pivot_points["General"]["count rebound"]
        self.__stats_pivot_points["Resistance"]["pct resistance no rebound"] = 100. * count_resistance_no_rebound / self.__stats_pivot_points["General"]["count no rebound"]
        self.__stats_pivot_points["Resistance"]["pct point"] = 100. * count_resistance_total / self.__stats_pivot_points["General"]["count points"]
        self.__stats_pivot_points["Resistance"]["pct perfect rebound"] = 100. * count_resistance_perfect_rebound / count_resistance_total
        self.__stats_pivot_points["Resistance"]["pct deep rebound"] = 100. * count_resistance_deep_rebound / count_resistance_total
        self.__stats_pivot_points["Resistance"]["pct rebound"] = 100. * (
                    count_resistance_perfect_rebound + count_resistance_deep_rebound) / count_resistance_total
        self.__stats_pivot_points["Resistance"]["pct no rebound"] = 100. * count_resistance_no_rebound / count_resistance_total
        self.__stats_pivot_points["Resistance"]["pt deep rebound"] = [self.__es_price_df.iloc[i][self.__KEY_PT_DEEP_REBOUND]
                                                                      for i in rows_deep_rebound if self.__es_price_df.iloc[i][
                                                                          self.__KEY_NO_PERFECT_REBOUND] == "R"]

        cdf_deep_rebound = self.__calc_cpf(pt_deep_rebound, int(max(pt_deep_rebound)))
        self.__stats_deep_rebound["General"][" "] = "Demark"
        self.__stats_deep_rebound["General"]["% of deep rebounds"] = 100
        self.__stats_deep_rebound["General"]["mean"] = np.mean(pt_deep_rebound)
        for i, x in enumerate(cdf_deep_rebound["x"]):
            self.__stats_deep_rebound["General"][str(int(x))] = cdf_deep_rebound["cpf"][i]

        cdf_deep_rebound_pivot = self.__calc_cpf(pt_deep_rebound_pivot, int(max(pt_deep_rebound)))
        self.__stats_deep_rebound["Pivot"][" "] = "pivot"
        self.__stats_deep_rebound["Pivot"]["% of deep rebounds"] = 100. * len(pt_deep_rebound_pivot) / len(pt_deep_rebound)
        self.__stats_deep_rebound["Pivot"]["mean"] = np.mean(pt_deep_rebound_pivot)
        for i, x in enumerate(cdf_deep_rebound_pivot["x"]):
            self.__stats_deep_rebound["Pivot"][str(int(x))] = cdf_deep_rebound_pivot["cpf"][i]

        cdf_deep_rebound_support = self.__calc_cpf(pt_deep_rebound_support, int(max(pt_deep_rebound)))
        self.__stats_deep_rebound["Support"][" "] = "support"
        self.__stats_deep_rebound["Support"]["% of deep rebounds"] = 100. * len(pt_deep_rebound_support) / len(
            pt_deep_rebound)
        self.__stats_deep_rebound["Support"]["mean"] = np.mean(pt_deep_rebound_support)
        for i, x in enumerate(cdf_deep_rebound_support["x"]):
            self.__stats_deep_rebound["Support"][str(int(x))] = cdf_deep_rebound_support["cpf"][i]

        cdf_deep_rebound_resistance = self.__calc_cpf(pt_deep_rebound_resistance, int(max(pt_deep_rebound)))
        self.__stats_deep_rebound["Resistance"][" "] = "resistance"
        self.__stats_deep_rebound["Resistance"]["% of deep rebounds"] = 100. * len(pt_deep_rebound_resistance) / len(
            pt_deep_rebound)
        self.__stats_deep_rebound["Resistance"]["mean"] = np.mean(pt_deep_rebound_resistance)
        for i, x in enumerate(cdf_deep_rebound_resistance["x"]):
            self.__stats_deep_rebound["Resistance"][str(int(x))] = cdf_deep_rebound_resistance["cpf"][i]

        # Analysis no rebound
        trend_no_rebound = [self.__es_price_df.iloc[i][self.__KEY_TREND] for i in rows_no_rebound]
        es_pp_no_rebound = [self.__es_price_df.iloc[i][self.__KEY_ES_PP] for i in rows_no_rebound]
        count_days_no_rebound = len(rows_no_rebound)
        self.__stats_no_rebound["Range"][" "] = " "
        self.__stats_no_rebound["Range"]["% uptrend"] = 100. * trend_no_rebound.count(1) / count_days_no_rebound
        self.__stats_no_rebound["Range"]["% downtrend"] = 100. * trend_no_rebound.count(-1) / count_days_no_rebound
        self.__stats_no_rebound["Range"]["% range"] = 100. * trend_no_rebound.count(0) / count_days_no_rebound
        self.__stats_no_rebound["ES PP"][" "] = " "
        self.__stats_no_rebound["ES PP"]["% es<PP (8:30)"] = 100. * (
                es_pp_no_rebound.count(0) + es_pp_no_rebound.count(2)) / count_days_no_rebound
        self.__stats_no_rebound["ES PP"]["% es>PP (8:30)"] = 100. * (
                    es_pp_no_rebound.count(1) + es_pp_no_rebound.count(-1)) / count_days_no_rebound

        self.__stats_no_rebound["Point"][" "] = " "
        self.__stats_no_rebound["Point"]["% resistance"] = 100. * point_no_rebound.count("R") / count_days_no_rebound
        self.__stats_no_rebound["Point"]["% pivot"] = 100. * point_no_rebound.count("PP") / count_days_no_rebound
        self.__stats_no_rebound["Point"]["% support"] = 100. * point_no_rebound.count("S") / count_days_no_rebound

    def __analyze_reversal(self):
        self.__analyze_general_reversal()
        self.__analyze_reversal_each_trend()
        self.__analyze_reversal_es_start_pivot()

    def __analyze_general_reversal(self):
        self.__es_price_with_reversal_df = self.__es_price_df[self.__es_price_df['Hour trend change'].notnull()]
        self.__stats_reversal["General"]["count days with reversal"] = len(self.__es_price_with_reversal_df.index)
        self.__stats_reversal["General"]["pct with reversal"] = 100. * self.__stats_reversal["General"]["count days with reversal"] / self.__num_days

    def __analyze_reversal_each_trend(self):
        # filter days for uptrend, downtrend and trade range
        self.__es_uptrend_df = self.__es_price_df[self.__es_price_df['Trend since 8:30'] == 1]
        self.__num_days_uptrend = len(self.__es_uptrend_df.index)
        self.__es_downtrend_df = self.__es_price_df[self.__es_price_df['Trend since 8:30'] == -1]
        self.__num_days_downtrend = len(self.__es_downtrend_df.index)
        self.__es_range_df = self.__es_price_df[self.__es_price_df['Trend since 8:30'] == 0]
        self.__num_days_range = len(self.__es_range_df.index)
        hour_reversal = self.__es_price_df[self.__es_price_df["New trend"].notnull()]["Hour trend change"].values
        cpf_hour_reversal = self.__calc_cpf_time(hour_reversal)
        self.__stats_hour_reversal["General"][" "] = "any"
        for i, t in enumerate(cpf_hour_reversal["x"]):
            self.__stats_hour_reversal["General"]["% " + t.strftime('%H:%M')] = cpf_hour_reversal["cpf"][i]

        # filter days with reversal according to trend
        self.__es_uptrend_then_reversal_df = self.__es_uptrend_df[self.__es_uptrend_df['Hour trend change'].notnull()]
        es_uptrend_then_downtrend_df = self.__es_uptrend_then_reversal_df[self.__es_uptrend_then_reversal_df["New trend"] == -1]
        self.__es_downtrend_then_reversal_df = self.__es_downtrend_df[self.__es_downtrend_df['Hour trend change'].notnull()]
        es_downtrend_then_uptrend_df = self.__es_downtrend_then_reversal_df[
            self.__es_downtrend_then_reversal_df["New trend"] == 1]
        self.__es_range_then_reversal_df = self.__es_range_df[self.__es_range_df['Hour trend change'].notnull()]
        es_range_then_uptrend_df = self.__es_range_then_reversal_df[self.__es_range_then_reversal_df["New trend"] == 1]
        es_range_then_downtrend_df = self.__es_range_then_reversal_df[self.__es_range_then_reversal_df["New trend"] == -1]
        self.__pct_uptrend_then_reversal = 100. * len(self.__es_uptrend_then_reversal_df.index) / self.__num_days_uptrend
        self.__pct_downtrend_then_reversal = 100. * len(self.__es_downtrend_then_reversal_df.index) / self.__num_days_downtrend
        self.__pct_range_then_reversal = 100. * len(self.__es_range_then_reversal_df.index) / self.__num_days_range

        # save stats
        self.__stats_reversal["reversal day"][" "] = str(self.__stats_reversal["General"]["count days with reversal"]) + " days with reversal"
        self.__stats_reversal["reversal day"]["% down then up"] = 100. * len(es_downtrend_then_uptrend_df.index) / self.__stats_reversal["General"]["count days with reversal"]
        self.__stats_reversal["reversal day"]["% up then down"] = 100. * len(es_uptrend_then_downtrend_df.index) / \
                                                                self.__stats_reversal["General"][
                                                                    "count days with reversal"]
        self.__stats_reversal["reversal day"]["% range then up"] = 100. * len(es_range_then_uptrend_df.index) / \
                                                                self.__stats_reversal["General"][
                                                                    "count days with reversal"]
        self.__stats_reversal["reversal day"]["% range then down"] = 100. * len(es_range_then_downtrend_df.index) / \
                                                                self.__stats_reversal["General"][
                                                                    "count days with reversal"]

        # uptrend day
        count_uptrend_days = len(self.__es_uptrend_df.index)
        range_uptrend_no_reversal_days = self.__es_uptrend_df[self.__es_uptrend_df["New trend"].isnull()][
            "Max Range 8:30 - 13"].values
        range_uptrend_with_reversal_days = self.__es_uptrend_df[self.__es_uptrend_df["New trend"].notnull()][
            "Max Range 8:30 - 13"].values
        self.__stats_reversal["uptrend day"][" "] = str(count_uptrend_days) + " uptrend days"
        self.__stats_reversal["uptrend day"]["% with reversal*"] = 100. * len(self.__es_uptrend_then_reversal_df.index) / count_uptrend_days
        uptrend_then_reversal_es_higher_pp = self.__es_uptrend_then_reversal_df[
            (self.__es_uptrend_then_reversal_df['ES & PP'] == 1) | (self.__es_uptrend_then_reversal_df['ES & PP'] == -1)]
        uptrend_then_reversal_es_lower_pp = self.__es_uptrend_then_reversal_df[
            (self.__es_uptrend_then_reversal_df['ES & PP'] == 0) | (self.__es_uptrend_then_reversal_df['ES & PP'] == 2)]
        self.__stats_reversal["uptrend day"]["% reversal when ES<PP at 8:30am**"] = 100. * len(uptrend_then_reversal_es_lower_pp.index) / len(self.__es_uptrend_then_reversal_df.index)
        self.__stats_reversal["uptrend day"]["% reversal when ES>PP at 8:30am**"] = 100. * len(uptrend_then_reversal_es_higher_pp.index) / len(self.__es_uptrend_then_reversal_df.index)
        self.__stats_reversal["uptrend day"]["avg. range when no reversal [pt]"] = np.mean(range_uptrend_no_reversal_days)
        self.__stats_reversal["uptrend day"]["avg. range when reversal [pt]"] = np.mean(
            range_uptrend_with_reversal_days)
        hour_reversal_uptrend = self.__es_uptrend_df[self.__es_uptrend_df["New trend"].notnull()]["Hour trend change"].values
        cpf_hour_reversal_uptrend = self.__calc_cpf_time(hour_reversal_uptrend)
        self.__stats_hour_reversal["Uptrend"][" "] = "uptrend"
        for i, t in enumerate(cpf_hour_reversal_uptrend["x"]):
            self.__stats_hour_reversal["Uptrend"]["% " + t.strftime('%H:%M')] = cpf_hour_reversal_uptrend["cpf"][i]

        # downtrend day
        count_downtrend_days = len(self.__es_downtrend_df.index)
        range_downtrend_no_reversal_days = self.__es_downtrend_df[self.__es_downtrend_df["New trend"].isnull()]["Max Range 8:30 - 13"].values
        range_downtrend_with_reversal_days = self.__es_downtrend_df[self.__es_downtrend_df["New trend"].notnull()]["Max Range 8:30 - 13"].values
        self.__stats_reversal["downtrend day"][" "] = str(count_downtrend_days) + " downtrend days"
        self.__stats_reversal["downtrend day"]["% with reversal*"] = 100. * len(
            self.__es_downtrend_then_reversal_df.index) / count_downtrend_days
        downtrend_then_reversal_es_higher_pp = self.__es_downtrend_then_reversal_df[
            (self.__es_downtrend_then_reversal_df['ES & PP'] == 1) | (self.__es_downtrend_then_reversal_df['ES & PP'] == -1)]
        downtrend_then_reversal_es_lower_pp = self.__es_downtrend_then_reversal_df[
            (self.__es_downtrend_then_reversal_df['ES & PP'] == 0) | (self.__es_downtrend_then_reversal_df['ES & PP'] == 2)]
        self.__stats_reversal["downtrend day"]["% reversal when ES<PP at 8:30am**"] = 100. * len(
            downtrend_then_reversal_es_lower_pp.index) / len(self.__es_downtrend_then_reversal_df.index)
        self.__stats_reversal["downtrend day"]["% reversal when ES>PP at 8:30am**"] = 100. * len(
            downtrend_then_reversal_es_higher_pp.index) / len(self.__es_downtrend_then_reversal_df.index)
        self.__stats_reversal["downtrend day"]["avg. range when no reversal [pt]"] = np.mean(range_downtrend_no_reversal_days)
        self.__stats_reversal["downtrend day"]["avg. range when reversal [pt]"] = np.mean(
            range_downtrend_with_reversal_days)
        hour_reversal_downtrend = self.__es_downtrend_df[self.__es_downtrend_df["New trend"].notnull()]["Hour trend change"].values
        cpf_hour_reversal_downtrend = self.__calc_cpf_time(hour_reversal_downtrend)
        self.__stats_hour_reversal["Downtrend"][" "] = "downtrend"
        for i, t in enumerate(cpf_hour_reversal_downtrend["x"]):
            self.__stats_hour_reversal["Downtrend"]["% " + t.strftime('%H:%M')] = cpf_hour_reversal_downtrend["cpf"][i]

        # range day
        count_range_days = len(self.__es_range_df.index)
        range_range_no_reversal_days = self.__es_range_df[self.__es_range_df["New trend"].isnull()]["Max Range 8:30 - 13"].values
        range_range_with_reversal_days = self.__es_range_df[self.__es_range_df["New trend"].notnull()]["Max Range 8:30 - 13"].values
        count_range_with_reversal = len(range_range_with_reversal_days)
        range_then_uptrend_days_df = self.__es_range_df[self.__es_range_df["New trend"] == 1]
        range_then_uptrend_range = range_then_uptrend_days_df["Max Range 8:30 - 13"].values
        range_then_downtrend_days_df = self.__es_range_df[self.__es_range_df["New trend"] == -1]
        range_then_downtrend_range = range_then_downtrend_days_df["Max Range 8:30 - 13"].values
        self.__range_then_reversal[" "] = str(count_range_with_reversal) + " range with reversal days"
        self.__range_then_reversal["% uptrend reversal"] = 100. * len(range_then_uptrend_days_df.index) / count_range_with_reversal
        self.__range_then_reversal["% downtrend reversal"] = 100. * len(range_then_downtrend_days_df.index) / count_range_with_reversal
        self.__range_then_reversal["avg. pt range no reversal"] = np.mean(range_range_no_reversal_days)
        self.__range_then_reversal["avg. pt range then uptrend"] = np.mean(range_then_uptrend_range)
        self.__range_then_reversal["avg. pt range then downtrend"] = np.mean(range_then_downtrend_range)

        self.__stats_reversal["range day"][" "] = str(count_range_days) + " range days"
        self.__stats_reversal["range day"]["% with reversal*"] = 100. * len(self.__es_range_then_reversal_df.index) / count_range_days
        range_then_reversal_es_higher_pp = self.__es_range_then_reversal_df[
            (self.__es_range_then_reversal_df['ES & PP'] == 1) | (self.__es_range_then_reversal_df['ES & PP'] == -1)]
        range_then_reversal_es_lower_pp = self.__es_range_then_reversal_df[
            (self.__es_range_then_reversal_df['ES & PP'] == 0) | (self.__es_range_then_reversal_df['ES & PP'] == 2)]
        self.__stats_reversal["range day"]["% reversal when ES<PP at 8:30am**"] = 100. * len(range_then_reversal_es_lower_pp.index) / len(self.__es_downtrend_then_reversal_df.index)
        self.__stats_reversal["range day"]["% reversal when ES>PP at 8:30am**"] = 100. * len(
            range_then_reversal_es_higher_pp.index) / len(self.__es_range_then_reversal_df.index)
        self.__stats_reversal["range day"]["avg. range when no reversal [pt]"] = np.mean(range_range_no_reversal_days)
        self.__stats_reversal["range day"]["avg. range when reversal [pt]"] = np.mean(range_range_with_reversal_days)
        hour_reversal_range = self.__es_range_df[self.__es_range_df["New trend"].notnull()]["Hour trend change"].values
        cpf_hour_reversal_range = self.__calc_cpf_time(hour_reversal_range)
        self.__stats_hour_reversal["Range"][" "] = "range"
        for i, t in enumerate(cpf_hour_reversal_range["x"]):
            self.__stats_hour_reversal["Range"]["% " + t.strftime('%H:%M')] = cpf_hour_reversal_range["cpf"][i]

    def __analyze_reversal_es_start_pivot(self):
        # filter days ES > PP or ES < PP at 8:30am
        self.__es_higher_pp_start = self.__es_price_df[(self.__es_price_df['ES & PP'] == 1) | (self.__es_price_df['ES & PP'] == -1)]
        self.__num_days_es_higher_pp_start = len(self.__es_higher_pp_start.index)
        self.__es_lower_pp_start = self.__es_price_df[(self.__es_price_df['ES & PP'] == 0) | (self.__es_price_df['ES & PP'] == 2)]
        self.__num_days_es_lower_pp_start = len(self.__es_lower_pp_start.index)

        # filter for days with reversals
        self.__es_higher_pp_start_then_reversal_df = self.__es_higher_pp_start[self.__es_higher_pp_start['Hour trend change'].notnull()]
        self.__es_lower_pp_start_then_reversal_df = self.__es_lower_pp_start[self.__es_lower_pp_start['Hour trend change'].notnull()]

        self.__pct_es_higher_pp_start_then_reversal = 100. * len(self.__es_higher_pp_start_then_reversal_df.index) / self.__num_days_es_higher_pp_start
        self.__pct_es_lower_pp_start_then_reversal = 100. * len(self.__es_lower_pp_start_then_reversal_df.index) / self.__num_days_es_lower_pp_start

    def __analyze_range(self):
        """
        Analyze range of each trend.
        """

        # Trading range day
        self.__stats_range_range["Range"][" "] = "range"
        self.__stats_range_range["Range ES<PP"][" "] = "range ES<PP 8:30"
        self.__stats_range_range["Range ES>PP"][" "] = "range ES>PP 8:30"
        range_range_all_days = self.__es_range_df["Max Range 8:30 - 13"].values
        range_range_es_lower_pp = self.__es_range_df[(self.__es_range_df["ES & PP"] == 1) | (self.__es_range_df["ES & PP"] == -1)]["Max Range 8:30 - 13"].values
        range_range_es_higher_pp = self.__es_range_df[(self.__es_range_df["ES & PP"] == 0) | (self.__es_range_df["ES & PP"] == 2)]["Max Range 8:30 - 13"].values
        range_cpf = self.__calc_cpf(range_range_all_days, int(max(range_range_all_days)), self.__BIN_RANGE_CPF)
        range_es_lower_pp_cpf = self.__calc_cpf(range_range_es_lower_pp, int(max(range_range_all_days)), self.__BIN_RANGE_CPF)
        range_es_higher_pp_cpf = self.__calc_cpf(range_range_es_higher_pp, int(max(range_range_all_days)), self.__BIN_RANGE_CPF)
        for i, p in enumerate(range_cpf["x"]):
            self.__stats_range_range["Range"]["% " + str(p) + "pt"] = range_cpf["cpf"][i]
            self.__stats_range_range["Range ES<PP"]["% " + str(p) + "pt"] = range_es_lower_pp_cpf["cpf"][i]
            self.__stats_range_range["Range ES>PP"]["% " + str(p) + "pt"] = range_es_higher_pp_cpf["cpf"][i]

        # Uptrend day
        self.__stats_range_uptrend["Uptrend"][" "] = "uptrend"
        self.__stats_range_uptrend["Uptrend ES<PP"][" "] = "uptrend ES<PP 8:30"
        self.__stats_range_uptrend["Uptrend ES>PP"][" "] = "uptrend ES>PP 8:30"
        range_uptrend_all_days = self.__es_uptrend_df["Max Range 8:30 - 13"].values
        range_uptrend_es_lower_pp = self.__es_uptrend_df[(self.__es_uptrend_df["ES & PP"] == 1) | (self.__es_uptrend_df["ES & PP"] == -1)][
            "Max Range 8:30 - 13"].values
        range_uptrend_es_higher_pp = self.__es_uptrend_df[(self.__es_uptrend_df["ES & PP"] == 0) | (self.__es_uptrend_df["ES & PP"] == 2)][
            "Max Range 8:30 - 13"].values
        uptrend_cpf = self.__calc_cpf(range_uptrend_all_days, int(max(range_uptrend_all_days)), self.__BIN_TREND_CPF)
        uptrend_es_lower_pp_cpf = self.__calc_cpf(range_uptrend_es_lower_pp, int(max(range_uptrend_all_days)), self.__BIN_TREND_CPF)
        uptrend_es_higher_pp_cpf = self.__calc_cpf(range_uptrend_es_higher_pp, int(max(range_uptrend_all_days)), self.__BIN_TREND_CPF)
        for i, p in enumerate(uptrend_cpf["x"]):
            self.__stats_range_uptrend["Uptrend"]["% " + str(p) + "pt"] = uptrend_cpf["cpf"][i]
            self.__stats_range_uptrend["Uptrend ES<PP"]["% " + str(p) + "pt"] = uptrend_es_lower_pp_cpf["cpf"][i]
            self.__stats_range_uptrend["Uptrend ES>PP"]["% " + str(p) + "pt"] = uptrend_es_higher_pp_cpf["cpf"][i]

        # Downtrend day
        self.__stats_range_downtrend["Downtrend"][" "] = "downtrend"
        self.__stats_range_downtrend["Downtrend ES<PP"][" "] = "downtrend ES<PP 8:30"
        self.__stats_range_downtrend["Downtrend ES>PP"][" "] = "downtrend ES>PP 8:30"
        range_downtrend_all_days = self.__es_downtrend_df["Max Range 8:30 - 13"].values
        range_downtrend_es_lower_pp = self.__es_downtrend_df[(self.__es_downtrend_df["ES & PP"] == 1) | (self.__es_downtrend_df["ES & PP"] == -1)][
            "Max Range 8:30 - 13"].values
        range_downtrend_es_higher_pp = self.__es_downtrend_df[(self.__es_downtrend_df["ES & PP"] == 0) | (self.__es_downtrend_df["ES & PP"] == 2)][
            "Max Range 8:30 - 13"].values
        downtrend_cpf = self.__calc_cpf(range_downtrend_all_days, int(max(range_downtrend_all_days)), self.__BIN_TREND_CPF)
        downtrend_es_lower_pp_cpf = self.__calc_cpf(range_downtrend_es_lower_pp, int(max(range_downtrend_all_days)), self.__BIN_TREND_CPF)
        downtrend_es_higher_pp_cpf = self.__calc_cpf(range_downtrend_es_higher_pp, int(max(range_downtrend_all_days)), self.__BIN_TREND_CPF)
        for i, p in enumerate(downtrend_cpf["x"]):
            self.__stats_range_downtrend["Downtrend"]["% " + str(p) + "pt"] = downtrend_cpf["cpf"][i]
            self.__stats_range_downtrend["Downtrend ES<PP"]["% " + str(p) + "pt"] = downtrend_es_lower_pp_cpf["cpf"][i]
            self.__stats_range_downtrend["Downtrend ES>PP"]["% " + str(p) + "pt"] = downtrend_es_higher_pp_cpf["cpf"][i]

    def __analyze_no_rebound(self):

        no_rebounds_df = self.__es_price_df[self.__es_price_df[self.__KEY_PT_DEEP_REBOUND].isnull() & self.__es_price_df[
            self.__KEY_NO_PERFECT_REBOUND].notnull()]
        no_rebounds_pivot_df = no_rebounds_df[no_rebounds_df["No perfect rebound"] =="PP"]
        no_rebounds_support_df = no_rebounds_df[no_rebounds_df["No perfect rebound"] == "S"]
        no_rebounds_resistance_df = no_rebounds_df[no_rebounds_df["No perfect rebound"] == "R"]

        cpf_hour_no_rebound_pivot = self.__calc_cpf_time(no_rebounds_pivot_df["Time cross"].values)
        cpf_hour_no_rebound_support = self.__calc_cpf_time(no_rebounds_support_df["Time cross"].values)
        cpf_hour_no_rebound_resistance = self.__calc_cpf_time(no_rebounds_resistance_df["Time cross"].values)

        self.__time_stats_no_rebound["Pivot"][" "] = "pivot"
        self.__time_stats_no_rebound["Support"][" "] = "support"
        self.__time_stats_no_rebound["Resistance"][" "] = "resistance"
        for i, t in enumerate(cpf_hour_no_rebound_pivot["x"]):
            self.__time_stats_no_rebound["Pivot"]["% " + t.strftime('%H:%M')] = cpf_hour_no_rebound_pivot["cpf"][i]
            self.__time_stats_no_rebound["Support"]["% " + t.strftime('%H:%M')] = cpf_hour_no_rebound_support["cpf"][i]
            self.__time_stats_no_rebound["Resistance"]["% " + t.strftime('%H:%M')] = cpf_hour_no_rebound_resistance["cpf"][i]

        cpf_body_no_rebound_pivot = self.__calc_cpf(no_rebounds_pivot_df[no_rebounds_pivot_df["Body candle"] > 0]["Body candle"].values.astype(int),
                                                                                              int(max(no_rebounds_df["Body "
                                                                                                                                "candle"].values)), 5)
        cpf_body_no_rebound_support = self.__calc_cpf(no_rebounds_support_df[no_rebounds_support_df["Body candle"] > 0]["Body candle"].values.astype(
            int), int(max(no_rebounds_df["Body "
                                                                                                                                       "candle"].values)), 5)
        cpf_body_no_rebound_resistance = self.__calc_cpf(no_rebounds_resistance_df[no_rebounds_resistance_df["Body candle"] > 0]["Body "
                                                                                                                          "candle"].values.astype(int), int(max(no_rebounds_df["Body "
                                                                                                                                             "candle"].values)), 5)
        self.__body_stats_no_rebound["Pivot"][" "] = "pivot"
        self.__body_stats_no_rebound["Support"][" "] = "support"
        self.__body_stats_no_rebound["Resistance"][" "] = "resistance"
        for i, t in enumerate(cpf_body_no_rebound_resistance["x"]):
            self.__body_stats_no_rebound["Pivot"]["% " + str(t) + "pt"] = cpf_body_no_rebound_pivot["cpf"][i]
            self.__body_stats_no_rebound["Support"]["% " + str(t) + "pt"] = cpf_body_no_rebound_support["cpf"][i]
            self.__body_stats_no_rebound["Resistance"]["% " + str(t)+"pt"] = cpf_body_no_rebound_resistance["cpf"][i]

        # TO DO:
        # stats PP: pct each pattern
        # stats PP: if range, pct up vs down
        # stats PP: if range PP, pct up vs down
        # stats all: cpf body candle
        # stats all: avg. pt rebound if candle body > 4 vs 0pt body
        # stats all: pct body > 0 vs body == 0

    def __write_html(self):
        """
        Write output file with all the statistics.
        """

        try:
            with open(os.path.expanduser(self.__PATH_FINAL_REPORT), 'w') as fo:
                fo.write("<html>\n<head>\n<title> \nOutput Data in an HTML file \
                          </title>\n</head> <body><h1><center>" + "ES price data from 8:30am to 3pm </center></h1>\n</body></html>")
                fo.write('<br/>')
                fo.write("<br/>Period: " + self._EsPriceAnalysis__es_price_df["Date"].iloc[0].strftime('%d/%m/%Y') + " to "
                         + self._EsPriceAnalysis__es_price_df["Date"].iloc[-1].strftime('%d/%m/%Y'))
                fo.write('<br/><br/>')

                # ================================ Analysis of reversals =================================
                fo.write("<center><b>Reversals (8:30am to 3pm)</b></center>")
                fo.write("<br/>Days with trend reversal: " + str(self.__round(self.__stats_reversal["General"]["pct with reversal"])) + "%")
                fo.write("<br/>")

                # pct days with a reversal in the morning
                days_reversal_df = pd.DataFrame([self.__stats_reversal["reversal day"]])
                days_reversal_df.set_index(" ", inplace=True)
                days_reversal_df.index.name = None
                fo.write(days_reversal_df.to_html().replace('<td>', '<td align="center">'))
                fo.write("*Trend to range are not showed in the table.<br/>")
                fo.write("<br/>")

                # range and ES vs. PP at 8:30am analysis
                fo.write("<br/>Analysis of range and ES vs. PP at 8:30am when a reversal occurs.")
                trend_days_reversal_df = pd.DataFrame([self.__stats_reversal["uptrend day"],
                                                       self.__stats_reversal["downtrend day"],
                                                       self.__stats_reversal["range day"]])
                trend_days_reversal_df.set_index(" ", inplace=True)
                trend_days_reversal_df.index.name = None
                fo.write(trend_days_reversal_df.to_html().replace('<td>', '<td align="center">'))
                fo.write("*% calculated against the number of days with that trend type.<br/>")
                fo.write("**% calculated against the number of days with a reversal.<br/>")
                fo.write("<br/>")

                # cumulative probability of reversal hours
                fo.write("<br/>Analysis of the cumulative probability of the time of the reversal.")
                trend_days_hour_reversal_df = pd.DataFrame([self.__stats_hour_reversal["General"],
                                                            self.__stats_hour_reversal["Uptrend"],
                                                            self.__stats_hour_reversal["Downtrend"],
                                                            self.__stats_hour_reversal["Range"]])
                trend_days_hour_reversal_df.set_index(" ", inplace=True)
                trend_days_hour_reversal_df.index.name = None
                fo.write(trend_days_hour_reversal_df.to_html().replace('<td>', '<td align="center">'))
                fo.write("<br/>")

                fo.write("<br/>Analysis of the trading range with reversal.")
                trend_reversal_df = pd.DataFrame([self.__range_then_reversal])
                trend_reversal_df.set_index(" ", inplace=True)
                trend_reversal_df.index.name = None
                fo.write(trend_reversal_df.to_html().replace('<td>', '<td align="center">'))
                fo.write("<br/>")

                # ======================================= Demark pivot points analysis ======================================
                fo.write('<br/><br/><br/>')
                fo.write("<center><b>Demark points (8:30am to 1pm)</b></center>")
                fo.write("<br/>Definitions:")
                fo.write("<br/><u>Perfect rebound</u>: ES hits the Demark point +/-2pt and bounces from it of at least 4pts (Mastering take profit)")
                fo.write("<br/><u>Deep rebound</u>: ES crosses the Demark point more than 2pt ")
                fo.write("and reverts towards the point within maximum 15.75pt from the cross.")
                fo.write("<br/>     ES then crosses again the point of at least 4pt (take profit)")
                fo.write("<br/><u>Rebound</u>: ES hits or crosses the Demark point of no more than 15.75pt ")
                fo.write("and bounce back of more than 4pt from the point.")
                fo.write("<br/><u>Failed rebound</u>: ES crosses the Demark point of more than 16pt without a 4pt bounce.")
                fo.write('<br/><br/>')
                fo.write("<br/>Frequency of Demark pivot points:")
                frequency_point_df = pd.DataFrame([{
                    " ": str(self.__stats_pivot_points["General"]["count points"]) + " crossed Demark points",
                    "% resistance": self.__stats_pivot_points["Resistance"]["pct point"],
                    "% pivot": self.__stats_pivot_points["Pivot"]["pct point"],
                    "% support": self.__stats_pivot_points["Support"]["pct point"]
                }])
                frequency_point_df.set_index(" ", inplace=True)
                frequency_point_df.index.name = None
                fo.write(frequency_point_df.to_html().replace('<td>', '<td align="center">'))
                fo.write("<br/>")
                # each dictionary is a row of the pandas dataframe (and therefore of the html table)
                # the dictionaries must have the same keys, which are the html table column names
                days_df = pd.DataFrame([{
                    " ": str(self.__stats_pivot_points["General"]["total days"]) + " days",
                    "% days with Demark": self.__stats_pivot_points["General"]["pct days with demark"],
                    "% days without Demark": self.__stats_pivot_points["General"]["pct days no demark"],
                    "% days with a rebound*": self.__stats_pivot_points["General"]["pct days with rebound"],
                }])
                days_df.set_index(" ", inplace=True)
                days_df.index.name = None
                fo.write(days_df.to_html().replace('<td>', '<td align="center">'))
                fo.write("*Percentage against the total number of days analyzed.")

                fo.write('<br/><br/>')
                fo.write("<br/>Days with at least one Demark point crossed:")
                days_with_demark_df = pd.DataFrame([{
                    " ": str(self.__stats_pivot_points["Days with demark"]["count"]) + " days with Demark points",
                    "% with at least 1 rebound": self.__stats_pivot_points["Days with demark"]["with rebound"],
                    "% with at least 1 perfect rebound": self.__stats_pivot_points["Days with demark"]["with perfect rebound"],
                    "% with at least 1 deep rebound": self.__stats_pivot_points["Days with demark"]["with deep rebound"],
                    "% with at least 1 rebound without failed rebound": self.__stats_pivot_points["Days with demark"][
                        "only rebound"],
                    "% with at least 1 perfect rebound without failed rebound": self.__stats_pivot_points["Days with demark"]["only perfect rebound"],
                    "% with at least 1 deep rebound without failed rebound": self.__stats_pivot_points["Days with demark"]["only deep rebound"],
                    "% with 1 failed rebound": self.__stats_pivot_points["Days with demark"]["with no rebound"],
                    "% with 1 failed rebound and without rebounds": self.__stats_pivot_points["Days with demark"]["only no rebound"]
                }])
                days_with_demark_df.set_index(" ", inplace=True)
                days_with_demark_df.index.name = None
                fo.write(days_with_demark_df.to_html().replace('<td>', '<td align="center">'))

                fo.write('<br/><br/>')
                fo.write("<br/>Probability of a rebound (4/16pt win) when approaching a Demark point:")
                # each dictionary is a row of the pandas dataframe (and therefore of the html table)
                # the dictionaries must have the same keys, which are the html table column names
                dict_demark_point = {
                    " ": "any Demark",
                    "% rebound": self.__stats_pivot_points["General"]["pct rebound"],
                    "% perfect rebound": self.__stats_pivot_points["General"]["pct perfect rebound"],
                    "% deep rebound": self.__stats_pivot_points["General"]["pct deep rebound"],
                    "% no rebound": self.__stats_pivot_points["General"]["pct no rebound"]
                }
                dict_pivot = {
                    " ": "pivot",
                    "% rebound": self.__stats_pivot_points["Pivot"]["pct rebound"],
                    "% perfect rebound": self.__stats_pivot_points["Pivot"]["pct perfect rebound"],
                    "% deep rebound": self.__stats_pivot_points["Pivot"]["pct deep rebound"],
                    "% no rebound": self.__stats_pivot_points["Pivot"]["pct no rebound"]
                }
                dict_support = {
                    " ": "support",
                    "% rebound": self.__stats_pivot_points["Support"]["pct rebound"],
                    "% perfect rebound": self.__stats_pivot_points["Support"]["pct perfect rebound"],
                    "% deep rebound": self.__stats_pivot_points["Support"]["pct deep rebound"],
                    "% no rebound": self.__stats_pivot_points["Support"]["pct no rebound"]
                }
                dict_resistance = {
                    " ": "resistance",
                    "% rebound": self.__stats_pivot_points["Resistance"]["pct rebound"],
                    "% perfect rebound": self.__stats_pivot_points["Resistance"]["pct perfect rebound"],
                    "% deep rebound": self.__stats_pivot_points["Resistance"]["pct deep rebound"],
                    "% no rebound": self.__stats_pivot_points["Resistance"]["pct no rebound"]
                }
                point_type_df = pd.DataFrame([dict_resistance,
                                              dict_pivot,
                                              dict_support,
                                              dict_demark_point])
                point_type_df.set_index(" ", inplace=True)
                point_type_df.index.name = None
                fo.write(point_type_df.to_html().replace('<td>', '<td align="center">'))

                fo.write('<br/><br/>')
                fo.write("<br/>Cumulative probability of deep rebound points when approaching a Demark point:")
                deep_type_df = pd.DataFrame([self.__stats_deep_rebound["General"],
                                             self.__stats_deep_rebound["Resistance"],
                                             self.__stats_deep_rebound["Pivot"],
                                             self.__stats_deep_rebound["Support"]])
                deep_type_df.set_index(" ", inplace=True)
                deep_type_df.index.name = None
                fo.write(deep_type_df.to_html().replace('<td>', '<td align="center">'))

                # fo.write('<br/><br/>')
                # fo.write("<br/>Probability of not bouncing when approaching a Demark point:")
                # trend_no_rebound_df = pd.DataFrame([self.__stats_no_rebound["Range"]])
                # trend_no_rebound_df.set_index(" ", inplace=True)
                # trend_no_rebound_df.index.name = None
                # fo.write(trend_no_rebound_df.to_html().replace('<td>', '<td align="center">'))

                # es_pp_no_rebound_df = pd.DataFrame([self.__stats_no_rebound["ES PP"]])
                # es_pp_no_rebound_df.set_index(" ", inplace=True)
                # es_pp_no_rebound_df.index.name = None
                # fo.write(es_pp_no_rebound_df.to_html().replace('<td>', '<td align="center">'))

                # point_no_rebound_df = pd.DataFrame([self.__stats_no_rebound["Point"]])
                # point_no_rebound_df.set_index(" ", inplace=True)
                # point_no_rebound_df.index.name = None
                # fo.write(point_no_rebound_df.to_html().replace('<td>', '<td align="center">'))
                # fo.write("<br/>")

                # ======================================= Range analysis ======================================
                fo.write('<br/><br/><br/>')
                fo.write("<center><b>Range analysis</b></center>")
                fo.write("<br/>Cumulative probability of the maximum range if trading range day.")
                range_range_df = pd.DataFrame([self.__stats_range_range["Range"],
                                               self.__stats_range_range["Range ES<PP"],
                                               self.__stats_range_range["Range ES>PP"]])
                range_range_df.set_index(" ", inplace=True)
                range_range_df.index.name = None
                fo.write(range_range_df.to_html().replace('<td>', '<td align="center">'))
                fo.write("<br/>Cumulative probability of the maximum range if uptrend day.")
                range_uptrend_df = pd.DataFrame([self.__stats_range_uptrend["Uptrend"],
                                                 self.__stats_range_uptrend["Uptrend ES<PP"],
                                                 self.__stats_range_uptrend["Uptrend ES>PP"]])
                range_uptrend_df.set_index(" ", inplace=True)
                range_uptrend_df.index.name = None
                fo.write(range_uptrend_df.to_html().replace('<td>', '<td align="center">'))
                fo.write("<br/>Cumulative probability of the maximum range if downtrend day.")
                range_downtrend_df = pd.DataFrame([self.__stats_range_downtrend["Downtrend"],
                                                   self.__stats_range_downtrend["Downtrend ES<PP"],
                                                   self.__stats_range_downtrend["Downtrend ES>PP"]])
                range_downtrend_df.set_index(" ", inplace=True)
                range_downtrend_df.index.name = None
                fo.write(range_downtrend_df.to_html().replace('<td>', '<td align="center">'))
                fo.write("<br/>")

                # OLD
                dict_rebound = {
                    " ": "rebound",
                    "pivot [%]": self.__stats_pivot_points["Pivot"]["pct pivot rebound"],
                    "support [%]": self.__stats_pivot_points["Support"]["pct support rebound"],
                    "resistance [%]": self.__stats_pivot_points["Resistance"]["pct resistance rebound"]
                }
                dict_perfect_rebound = {
                    " ": "perfect rebound",
                    "pivot [%]": self.__stats_pivot_points["Pivot"]["pct pivot perfect rebound"],
                    "support [%]": self.__stats_pivot_points["Support"]["pct support perfect rebound"],
                    "resistance [%]": self.__stats_pivot_points["Resistance"]["pct resistance perfect rebound"]
                }
                dict_deep_rebound = {
                    " ": "deep rebound",
                    "pivot [%]": self.__stats_pivot_points["Pivot"]["pct pivot deep rebound"],
                    "support [%]": self.__stats_pivot_points["Support"]["pct support deep rebound"],
                    "resistance [%]": self.__stats_pivot_points["Resistance"]["pct resistance deep rebound"]
                }
                dict_no_rebound = {
                    " ": "no rebound",
                    "pivot [%]": self.__stats_pivot_points["Pivot"]["pct pivot no rebound"],
                    "support [%]": self.__stats_pivot_points["Support"]["pct support no rebound"],
                    "resistance [%]": self.__stats_pivot_points["Resistance"]["pct resistance no rebound"]
                }
                rebound_type_df = pd.DataFrame([dict_rebound,
                                                dict_perfect_rebound,
                                                dict_deep_rebound,
                                                dict_no_rebound])
                rebound_type_df.set_index(" ", inplace=True)
                rebound_type_df.index.name = None

                # =========================== No rebound analysis ========================

                no_rebound_time_analysis_df = pd.DataFrame([self.__time_stats_no_rebound["Pivot"],
                                                            self.__time_stats_no_rebound["Support"],
                                                            self.__time_stats_no_rebound["Resistance"]
                                                            ])
                no_rebound_time_analysis_df.set_index(" ", inplace=True)
                no_rebound_time_analysis_df.index.name = None
                fo.write(no_rebound_time_analysis_df.to_html().replace('<td>', '<td align="center">'))

                fo.write("<br/>")
                no_rebound_body_analysis_df = pd.DataFrame([self.__body_stats_no_rebound["Pivot"],
                                                            self.__body_stats_no_rebound["Support"],
                                                            self.__body_stats_no_rebound["Resistance"]
                                                            ])
                no_rebound_body_analysis_df.set_index(" ", inplace=True)
                no_rebound_body_analysis_df.index.name = None
                fo.write(no_rebound_body_analysis_df.to_html().replace('<td>', '<td align="center">'))


        except Exception as e:
            print('Cannot create the html file:', e)
            sys.exit(1)  # stop the main function with exit code 1

    @staticmethod
    def __calc_cpf(data: list, bin_max: int, bin_span: int = 2) -> dict:
        """Calculate the cumulative probability function."""

        cpf = []
        x_cpf = [p for p in range(bin_span, bin_max, bin_span)] + [bin_max]
        n = float(len(data))
        for x in x_cpf:
            cpf.append(100. * sum(i <= x for i in data) / n)
        return {"cpf": cpf, "x": x_cpf}

    def __calc_cpf_time(self, data: list) -> dict:
        """Calculate the cumulative probability function."""

        cpf = []
        n = float(len(data))
        for x in self.__x_cpf_time:
            cpf.append(100. * sum(i <= x for i in data) / n)
        return {"cpf": cpf, "x": self.__x_cpf_time}

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
