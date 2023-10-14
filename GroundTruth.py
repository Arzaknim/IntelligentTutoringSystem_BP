import pandas as pd


class GroundTruth:

    def __init__(self):
        self.gt = pd.read_csv('periodic_table.csv').iloc[:, [1, 2]].to_dict()
