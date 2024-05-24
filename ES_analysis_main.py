# Copyright (c) 2024 Jacopo Ventura

import pandas as pd
import timeit
from helper.ES_analysis import EsPriceAnalysis


# Graphical options for dataframe print
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
pd.options.display.float_format = '{:,.1f}'.format

ES_DATA_FILE = 'trading_journal.xlsx'
FOLDER = '~/Desktop/MasteringSP500/'


# TO INVESTIGATE:

# 2. Histogram hours + cumulative probability (from 15 to 8)
# stats ES rebound at PP, S , R
# 5. range distribution (histogram)
# range distribution if uptrend, downtrend, 0
# range distribution if ES < PP at 8.30; ES > PP at 8.30

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    t0 = timeit.default_timer()

    es = EsPriceAnalysis(FOLDER, ES_DATA_FILE)
    es.run()
    elapsed = timeit.default_timer() - t0
    print(f'Execution time: {elapsed:.2}s')

    # fig = px.histogram(es_price_with_reversal_df, x="Hour trend change")
    # fig.update_layout(bargap=0.2)
    # fig.show()
