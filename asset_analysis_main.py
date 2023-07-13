import datetime
import pandas
import timeit
import yfinance as yf
from helper.data_analysis import PriceAnalysis

# product-level code: https://www.zenesys.com/blog/python-coding-standards-best-practices
# structure: https://stackoverflow.com/questions/193161/what-is-the-best-project-structure-for-a-python-application

# Graphical options for dataframe print
pandas.set_option('display.width', 400)
pandas.set_option('display.max_columns', 10)
pandas.options.display.float_format = '{:,.1f}'.format

TICKER = "SPY"
PATH_TO_REPORT = '~/Desktop/Trading/'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    t0 = timeit.default_timer()

    # Query data from yahoo finance
    start = datetime.datetime(2022, 1, 1)  # Year, Month, Day
    end = datetime.datetime(2023, 2, 10)  # Year, Month, Day

    # Test download vix data from yahoo
    # data available from 02.01.1990
    # database: https://finance.yahoo.com/quote/%5EVIX/history?period1=631238400&period2=1689206400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
    data = yf.download('^VIX', start=datetime.datetime(1990, 1, 2), end=end)
    print(data.head())
    spy = PriceAnalysis(TICKER, start, end, PATH_TO_REPORT)
    spy.run()
    elapsed = timeit.default_timer() - t0
    print(f'Execution time: {elapsed:.2}s')
