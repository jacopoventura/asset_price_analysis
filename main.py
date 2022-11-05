import datetime
import timeit
from helper.data_analysis import PriceAnalysis

# TO DO:
# learn git rebase for versioning
# add month label xaxes plot weekly variation
# plot 23DTE change barplot
# set default input variable values
# handle weeks with 1 day
# optimize class
# make product-level code (Error handling; handle week with 1 day)
# https://stackoverflow.com/questions/193161/what-is-the-best-project-structure-for-a-python-application


TICKER = "SPY"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    t0 = timeit.default_timer()

    # Query data from yahoo finance
    start = datetime.datetime(2022, 1, 1)  # Year, Month, Day
    end = datetime.datetime.now()
    spy_2022 = PriceAnalysis(TICKER, start, end)
    spy_2022.run()
    elapsed = timeit.default_timer() - t0
    print(f'Time: {elapsed:.2}s')
