# Copyright (c) 2024 Jacopo Ventura

import datetime
import pandas
from helper.data_analysis import PriceAnalysis
import streamlit as st
import streamlit.components.v1 as components

# product-level code: https://www.zenesys.com/blog/python-coding-standards-best-practices
# structure: https://stackoverflow.com/questions/193161/what-is-the-best-project-structure-for-a-python-application

# Graphical options for dataframe print
pandas.set_option('display.width', 400)
pandas.set_option('display.max_columns', 10)
pandas.options.display.float_format = '{:,.1f}'.format

PATH_TO_REPORT = '~/Desktop/projects/Trading/'

path_to_html = "./"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # App title
    st.title("Price movement analysis")

    # App description

    # Input parameters of the app
    ticker = st.text_input("Ticker: ", value="SPY")
    date_start = st.date_input("Start date: ", value=datetime.datetime(2022, 1, 1))
    date_start = datetime.datetime(date_start.year, date_start.month, date_start.day)

    date_end = st.date_input("End date: ", value=datetime.datetime(2023, 1, 20))
    date_end = datetime.datetime(date_end.year, date_end.month, date_end.day)

    dte_long = st.number_input("Number of trading days of the sell put debit:", value=23)

    # Run analysis
    ticker_analysis = PriceAnalysis(ticker, date_start, date_end, dte_long, path_to_html)
    ticker_analysis.run()

    st.success('This is a success message!', icon="✅")

    # Download generated report
    with open(ticker_analysis.FILENAME, 'rb') as f:
        btn = st.download_button(
            label="Download report (HTML)",
            data=f,
            file_name=ticker_analysis.FILENAME,
        )

    # Show table in the app
    show_analysis = st.checkbox("Show analysis", value=False)

    if show_analysis:
        st.write("Great!")




