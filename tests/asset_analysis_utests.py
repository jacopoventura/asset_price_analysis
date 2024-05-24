# Copyright (c) 2024 Jacopo Ventura

import unittest
import datetime
import re
from helper.data_analysis import PriceAnalysis

dataset = {
    "positive": {
        "data": [1, 2, 3, 4, 5, 4, 1, 1, 0, 9],
        "bins":  [3, 6, 9],
        "cpf": [60., 90., 100.]
                },
    "negative": {
        "data": [-1, -2, -3, -4, -5, -4, -1, -1, 0, -9],
        "bins":  [-3, -6, -9],
        "cpf": [60., 90., 100.]
                },
    "positive_detailed": {
        "data": [1, 2, 3, 4, 5, 4, 1, 1, 0, 9],
        "bins": [1, 2, 3, 4],
        "cpf": [40., 50., 60., 80.]
    },
    "negative_detailed": {
        "data": [-1, -2, -3, -4, -5, -4, -1, -1, 0, -9],
        "bins": [-1, -2, -3, -4],
        "cpf": [40., 50., 60., 80.]
    },
}

start = datetime.datetime(2022, 1, 1)  # Year, Month, Day
end = datetime.datetime(2023, 2, 10)  # Year, Month, Day
spy = PriceAnalysis("SPY", start, end, "/tmp/")


# helper
def get_solution_from_dict(input_cpf: dict, input_bins: list) -> list:
    solution = []

    for bin_value in input_bins:
        for key in input_cpf.keys():
            n = re.findall("[\d\.\d]+", key)
            if n:
                if bin_value == float(n[0]):
                    solution.append(input_cpf[key])

    return solution


class TestCumulativeProbability(unittest.TestCase):

    def test_cpf_positive_data(self):
        cpf = spy._PriceAnalysis__calc_cpf(dataset["positive"]["data"], dataset["positive"]["bins"])
        self.assertEqual(dataset["positive"]["cpf"], cpf)

    def test_cpf_negative_data(self):
        cpf = spy._PriceAnalysis__calc_cpf([-i for i in dataset["negative"]["data"]], [-i for i in dataset["negative"]["bins"]])
        self.assertEqual(dataset["negative"]["cpf"], cpf)

    def test_cumulative_positive_data(self):
        cpf = spy._PriceAnalysis__calc_cumulative_probability(dataset["positive_detailed"]["data"])
        solution = get_solution_from_dict(cpf, dataset["positive_detailed"]["bins"])
        self.assertEqual(dataset["positive_detailed"]["cpf"], solution)

    def test_cumulative_negative_data(self):
        cpf = spy._PriceAnalysis__calc_cumulative_probability(dataset["negative_detailed"]["data"])
        solution = get_solution_from_dict(cpf, [-i for i in dataset["negative_detailed"]["bins"]])
        self.assertEqual(dataset["negative_detailed"]["cpf"], solution)


if __name__ == '__main__':
    unittest.main()
