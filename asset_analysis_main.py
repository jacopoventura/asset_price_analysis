import datetime
import pandas
import timeit
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
    start = datetime.datetime(2022, 9, 1)  # Year, Month, Day
    end = datetime.datetime(2023, 7, 20)  # Year, Month, Day
    spy = PriceAnalysis(TICKER, start, end, PATH_TO_REPORT)
    spy.run()
    elapsed = timeit.default_timer() - t0
    print(f'Execution time: {elapsed:.2}s')
