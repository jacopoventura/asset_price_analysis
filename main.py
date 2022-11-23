import datetime
import timeit
from helper.data_analysis import PriceAnalysis

# TO DO:
# make product-level code: https://www.zenesys.com/blog/python-coding-standards-best-practices
# add month label xaxes plot weekly variation (https://plotly.com/python/tick-formatting/
# plot 23DTE change barplot
# https://stackoverflow.com/questions/193161/what-is-the-best-project-structure-for-a-python-application


TICKER = "SPY"
PATH_TO_REPORT = '~/Desktop/MasteringSP500/'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    t0 = timeit.default_timer()

    # Query data from yahoo finance
    start = datetime.datetime(2022, 1, 1)  # Year, Month, Day
    end = datetime.datetime.now()  # Year, Month, Day
    spy = PriceAnalysis(TICKER, start, end, PATH_TO_REPORT)
    spy.run()
    elapsed = timeit.default_timer() - t0
    print(f'Execution time: {elapsed:.2}s')
