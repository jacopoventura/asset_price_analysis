import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import plotly.graph_objects as go

# Graphical options for dataframe print
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
pd.options.display.float_format = '{:,.2f}'.format


class PriceAnalysis:
    """
    Class to analyze historical pricing data of a specific asset.
    """

    def __init__(self, ticker, start, end):
        """
        Initialize the class PriceAnalysis with ticker, start and end time for the price analysis.
        :param ticker: ticker of the asset class
        :type ticker: str
        :param start: start date for the price analysis
        :type start:datetime
        :param end: end date for the price analysis
        :type end: datetime
        """

        self.__SOURCE = 'yahoo'

        self.__WEEK_TRADING_DAYS = 5
        self.__MONTH_TRADING_DAYS = 23
        self.__WEEK_MAX_CHANGE_PCT = 6
        self.__MONTH_MAX_CHANGE_PCT = 12
        self.__NUMBER_WEEKS_PER_YEAR = 52
        self.__DTE_LONG = 23

        self.__plot_column_width = 0.75

        self.__ticker = ticker
        self.__date_start = start
        self.__date_end = end
        self.__number_of_weeks = (end - start).days // 7

        self.__PATH_TO_HTML = '~/Desktop/MasteringSP500/'
        if start.year == end.year:
            self.__years_analysis = str(int(start.year))
        else:
            self.__years_analysis = str(int(start.year)) + '-' + str(int(end.year))
        self.__filename = self.__PATH_TO_HTML + self.__ticker + '_' + self.__years_analysis + '_update' + self.__date_end.strftime('%d%m%Y') + '.html'

        self.__price_history_df = None
        self.__weekly_change_monday_to_friday = None
        self.__weekly_change_friday_to_friday = None
        self.__weekly_change_df = None
        self.__weekly_change_monday_conditional_df = None
        self.__weekly_dte_change_df = None
        self.__monthly_dte_change_df = None

        self.__years_list = []
        self.__weekly_change_first_day_positive = []
        self.__weekly_change_first_day_negative = []
        self.__weekly_change_first_day_positive_week_count = []
        self.__weekly_change_first_day_negative_week_count = []

    def run(self):
        """
        Run the whole analysis and save the final document with all the statistics.
        """

        # Step 1: get historical price data for the selected time period
        self.query_price()

        # Step 2: calculate weekly statistics
        self.__calc_weekly_statistics()
        self.__calc_weekly_conditional_statistics()
        self.__weekly_dte_change_df = self.__calc_DTE_statistics(self.__WEEK_TRADING_DAYS,
                                                                 self.__WEEK_MAX_CHANGE_PCT)

        # Step 3: calculate monthly statistics
        self.__monthly_dte_change_df = self.__calc_DTE_statistics(self.__DTE_LONG,
                                                                  self.__MONTH_MAX_CHANGE_PCT)

        # Step 4: make html report
        self.__write_html()

    def update_analysis_period(self, start, end):
        """
        Update the time period for the analysis.
        :param start: start date for the price analysis
        :type start:datetime
        :param end: end date for the price analysis
        :type end: datetime
        """

        self.__date_start = start
        self.__date_end = end
        self.__number_of_weeks = (end - start).days // 7

        if start.year == end.year:
            self.__years_analysis = str(int(start.year))
        else:
            self.__years_analysis = str(int(start.year)) + '-' + str(int(end.year))
        self.__filename = self.__PATH_TO_HTML + self.__ticker + '_' + self.__years_analysis + '_update' + self.__date_end.strftime(
            '%d%m%Y') + '.html'

    def __write_html(self):
        """
        Write output file with all the statistics.
        """

        num_trading_days = int(len(self._PriceAnalysis__price_history_df.index))
        with open(
                os.path.expanduser(self.__filename),
                'w') as fo:
            fo.write("Statistiche " + self.__ticker + " " + self.__years_analysis)
            fo.write("<br/>Periodo: " + self.__date_start.strftime('%d/%m/%Y'))
            fo.write(" al "+ self.__date_end.strftime('%d/%m/%Y'))
            fo.write(" (" + str(int(self.__number_of_weeks)) + " settimane)")
            fo.write("<br/>Aggiornato al " + self.__date_end.strftime('%d/%m/%Y'))
            fo.write('<br/>' + "Tabelle della probabilita' cumulativa di variazione.")
            fo.write('<br/>' + '<br/>' + "Tabelle su variazione settimanale (Monday OPEN e Friday CLOSE)")
            fo.write('<br/>')
            fo.write(self.__weekly_change_df.to_html().replace('<td>', '<td align="center">'))
            fo.write('<br/>')
            fo.write(self.__weekly_dte_change_df.to_html().replace('<td>', '<td align="center">'))
            fo.write('<br/>')
            fo.write(self.__weekly_change_monday_conditional_df.to_html().replace('<td>', '<td align="center">'))
            fo.write('<br/>' + '<br/>' + "Tabelle su variazione " + str(self.__MONTH_TRADING_DAYS))
            fo.write(" DTE (giorni di trading effettivi, Daily OPEN)")
            fo.write("<br>" + str(num_trading_days) + " giorni analizzati (ultima OPEN il ")
            fo.write(self._PriceAnalysis__price_history_df["Date"][num_trading_days - self.__DTE_LONG].strftime('%d/%m/%Y') + ")")
            fo.write('<br/>')
            fo.write(self.__monthly_dte_change_df.to_html().replace('<td>', '<td align="center">'))
            fig = self.__make_plot_weekly_change()
            fo.write((fig.to_html(full_html=False, include_plotlyjs='cdn')))

    def __make_plot_weekly_change(self):
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
            go.Bar(name='positive (1st day +)',
                   x=week_positive_if_first_positive["week"],
                   y=week_positive_if_first_positive["change"],
                   marker_color='green',
                   marker_line_color='green',
                   width=self.__plot_column_width
                   ),
            go.Bar(name='positive (1st day -)',
                   x=week_positive_if_first_negative["week"],
                   y=week_positive_if_first_negative["change"],
                   marker_color='green',
                   marker_line_color='red',
                   marker_pattern_shape="/",
                   width=self.__plot_column_width
                   ),
            go.Bar(name='negative (1st day -)',
                   x=week_negative_if_first_negative["week"],
                   y=week_negative_if_first_negative["change"],
                   marker_color='red',
                   marker_line_color='red',
                   width=self.__plot_column_width
                   ),
            go.Bar(name='negative (1st day +)',
                   x=week_negative_if_first_positive["week"],
                   y=week_negative_if_first_positive["change"],
                   marker_color='red',
                   marker_line_color='green',
                   marker_pattern_shape="/",
                   width=self.__plot_column_width
                   )
        ])

        # Change the bar mode
        fig.update_layout(title="<br><b>Weekly change<b>",
                          barmode='group')
        fig.update_yaxes(title_text="change [%]")
        fig.update_xaxes(title_text="week count")

        return fig

    def query_price(self):
        """
        Query data of the ticker for the input timerange from the source database.
        """

        self.__price_history_df = web.DataReader(self.__ticker,
                                                 self.__SOURCE,
                                                 self.__date_start,
                                                 self.__date_end)
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
        self.__price_history_df = self.__price_history_df.reset_index(level=0)
        self.__years_list = list(set(years_list))  # get unique years

    def __calc_DTE_statistics(self, dte, max_change_pct):
        """
        Calculate statistics given for any day given a DTE.
        :param dte: date to end (effective trading days)
        :type dte: int
        :param max_change_pct: max weekly change in percentage expressed in the range 0-100
        :type max_change_pct: float
        :return: dataframe with the price change for the selected DTE
        :rtype: pd.dataframe
        """

        change_list = self.__calc_change_DTE(dte)
        change_positive, change_negative = self.__calc_distribution(change_list, max_change_pct)
        change_positive["Case"] = "Daily to " + str(int(dte)) + "DTE: positive"
        change_negative["Case"] = "Daily to " + str(int(dte)) + "DTE: negative"
        # noinspection PyTypeChecker
        change_df = pd.DataFrame.from_dict([change_positive, change_negative])
        change_df.set_index("Case", inplace=True)
        change_df.index.name = None
        return change_df

    def __calc_weekly_statistics(self):
        """
        Calculate the statistics of the weekly change in the asset price.
        """

        # Monday to Friday (first to last week days)
        weekly_change_monday_to_friday = self.__calc_weekly_movement()
        monday_to_friday_positive_dict, monday_to_friday_negative_dict = self.__calc_distribution(
            weekly_change_monday_to_friday,
            self.__WEEK_MAX_CHANGE_PCT)
        monday_to_friday_positive_dict["Case"] = "Monday to Friday: positive"
        monday_to_friday_negative_dict["Case"] = "Monday to Friday: negative"

        # Friday to friday (last to last week days)
        weekly_change_friday_to_friday = self.__calc_weekly_friday_to_friday_movement()
        friday_to_friday_positive_dict, friday_to_friday_negative_dict = self.__calc_distribution(
            weekly_change_friday_to_friday,
            self.__WEEK_MAX_CHANGE_PCT)
        friday_to_friday_positive_dict["Case"] = "Friday to Friday: positive"
        friday_to_friday_negative_dict["Case"] = "Friday to Friday: negative"

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
            self.__weekly_change_first_day_positive, self.__WEEK_MAX_CHANGE_PCT)
        weekly_positive_if_monday_negative_dict, weekly_negative_if_monday_negative_dict = self.__calc_distribution(
            self.__weekly_change_first_day_negative, self.__WEEK_MAX_CHANGE_PCT)

        weekly_positive_if_monday_positive_dict["Case"] = "Week if Monday positive: positive"
        weekly_negative_if_monday_positive_dict["Case"] = "Week if Monday positive: negative"
        weekly_positive_if_monday_negative_dict["Case"] = "Week if Monday negative: positive"
        weekly_negative_if_monday_negative_dict["Case"] = "Week if Monday negative: positive"
        # noinspection PyTypeChecker
        self.__weekly_change_monday_conditional_df = pd.DataFrame.from_dict([weekly_positive_if_monday_positive_dict,
                                                                             weekly_negative_if_monday_positive_dict,
                                                                             weekly_positive_if_monday_negative_dict,
                                                                             weekly_negative_if_monday_negative_dict])
        self.__weekly_change_monday_conditional_df.set_index("Case", inplace=True)
        self.__weekly_change_monday_conditional_df.index.name = None

    def __calc_number_of_weeks_in_year(self, year):
        """
        Get the number of available weeks for the selected year.
        :param year: year for the calculation of the number of weeks
        :type year: int
        :return: number of weeks considered in the input year
        :rtype int
        """

        return int(np.max(self.__price_history_df.loc[self.__price_history_df["Year"] == year]["Week number"]))

    def __calc_weekly_if_monday(self):
        """
        Calculate the weekly change depending on Monday's (or first weekday) change.
        """

        week_counter = 0
        for year in self.__years_list:
            number_of_weeks_in_the_year = self.__calc_number_of_weeks_in_year(year)
            for week_number in range(1, number_of_weeks_in_the_year + 1):
                week_counter += 1
                # filter the selected week and the previous week
                week_df = self.__price_history_df.loc[self.__price_history_df["Week number"] == week_number]
                first_day = np.min(week_df["Weekday"])
                last_day = np.max(week_df["Weekday"])
                week_open = week_df[week_df["Weekday"] == first_day].iloc[0]["Open"]
                week_close = week_df[week_df["Weekday"] == last_day].iloc[0]["Close"]
                week_change = 100.0 * (week_close - week_open) / week_open
                first_day_change = 100.0 * (
                            week_df[week_df["Weekday"] == first_day].iloc[0]["Close"] - week_open) / week_open
                if first_day_change > 0:
                    self.__weekly_change_first_day_positive.append(week_change)
                    self.__weekly_change_first_day_positive_week_count.append(week_counter)
                else:
                    self.__weekly_change_first_day_negative.append(week_change)
                    self.__weekly_change_first_day_negative_week_count.append(week_counter)

    def __calc_weekly_movement(self):
        """
        Calculate the weekly movement of the ticker.
        :return: list with the weekly changes
        :rtype: list
        """

        change_monday_to_friday_list = []
        for year in self.__years_list:
            number_of_weeks_in_the_year = self.__calc_number_of_weeks_in_year(year)
            for week_number in range(1, number_of_weeks_in_the_year + 1):
                # filter the selected week and the previous week
                week_df = self.__price_history_df.loc[self.__price_history_df["Week number"] == week_number]
                first_day = np.min(week_df["Weekday"])
                last_day = np.max(week_df["Weekday"])
                week_open = week_df[week_df["Weekday"] == first_day].iloc[0]["Open"]
                week_close = week_df[week_df["Weekday"] == last_day].iloc[0]["Close"]
                # calculate weekly changes
                change_monday_to_friday_list.append(100.0 * (week_close - week_open) / week_open)
        return change_monday_to_friday_list

    def __calc_weekly_friday_to_friday_movement(self):
        """
        Calculate the weekly movement of the ticker.
        :return: list with the weekly changes
        :rtype: list
        """

        change_friday_to_friday_list = []
        for year in self.__years_list:
            number_of_weeks_in_the_year = self.__calc_number_of_weeks_in_year(year)
            for week_number in range(2, number_of_weeks_in_the_year + 1):
                # filter the selected week and the previous week
                week_df = self.__price_history_df.loc[self.__price_history_df["Week number"] == week_number]
                previous_week_df = self.__price_history_df.loc[self.__price_history_df["Week number"] == week_number - 1]
                last_day = np.max(week_df["Weekday"])
                last_day_previous_week = np.max(previous_week_df["Weekday"])
                week_close = week_df[week_df["Weekday"] == last_day].iloc[0]["Close"]
                previous_week_close = previous_week_df[previous_week_df["Weekday"] == last_day_previous_week].iloc[0][
                    "Close"]
                # calculate weekly changes
                change_friday_to_friday_list.append(100.0 * (week_close - previous_week_close) / previous_week_close)
        return change_friday_to_friday_list

    def __calc_change_DTE(self, dte):
        """
        Calculate the price change (close to close) given an input DTE (Date To End) of every day of the price data.
        :param dte: Date To End (effective trading days until option expiration)
        :type dte: int
        :return: list with the DTE changes for each day of the selected period
        :rtype: list"""

        num_data = len(self.__price_history_df.index)
        change_list = []
        for idx in range(0, num_data - dte):
            start = self.__price_history_df.iloc[idx]["Close"]
            end = self.__price_history_df.iloc[idx + dte]["Close"]
            change_list.append(100.0 * (end - start) / start)
        return change_list

    @staticmethod
    def __calc_positive_negative_change_lists(change_list):
        """
        Split the input change list into positive change list and negative change list.
        :param change_list: list of price changes
        :type change_list: list
        :returns: lists of positive and negative changes
        :rtype: list
        """

        positive_change_list = sorted(list(filter(lambda change: change >= 0.0, change_list)))
        negative_change_list = sorted(list(filter(lambda change: change < 0.0, change_list)))
        return positive_change_list, negative_change_list

    def __calc_distribution(self, change_list, pct_max):
        """
        Calculate the distributions of positive and negative changes.
        Calculate the cumulative probability distribution.
        :param change_list: list of price changes
        :type change_list: list
        :param pct_max: max price change percentage in the range 0-100
        :type pct_max: float
        :returns: dictionaries of positive and negative changes with the cumulative distribution up to pct_max
        :rtype: dict"""

        positive_change, negative_change = self.__calc_positive_negative_change_lists(change_list)
        num_positive = len(positive_change)
        num_negative = len(negative_change)
        step = 1
        pct_list = np.linspace(0, pct_max, int((pct_max / step)) + 1)
        dict_positive = {"frequency [%]": 100.0 * num_positive / (num_positive + num_negative)}
        dict_negative = {"frequency [%]": 100.0 * num_negative / (num_positive + num_negative)}
        cumulative_positive = 0
        cumulative_negative = 0

        # calculate cumulative distribution
        for pct_low, pct_high in zip(pct_list, pct_list[1:]):
            pct_positive = 100.0 * len(list(filter(lambda x: pct_low <= x < pct_high, positive_change))) / num_positive
            cumulative_positive += pct_positive
            dict_positive[str(int(pct_high)) + "% change"] = cumulative_positive
            pct_negative = 100.0 * len(
                list(filter(lambda x: pct_low <= abs(x) < pct_high, negative_change))) / num_negative
            cumulative_negative += pct_negative
            dict_negative[str(int(pct_high)) + "% change"] = cumulative_negative
        dict_positive["max change [%]"] = np.max(positive_change)
        dict_positive["avg change [%]"] = np.mean(positive_change)
        dict_negative["max change [%]"] = np.min(negative_change)
        dict_negative["avg change [%]"] = np.mean(negative_change)

        return dict_positive, dict_negative

    def get_price_history(self):
        """
        Return price history dataframe.
        :return: dataframe of the price history of the asset
        :rtype: pd.dataframe
        """

        return self.__price_history_df

    def get_source(self):
        """
        Return source for price query.
        :return: price history query source
        :rtype: str
        """
        return self.__SOURCE
