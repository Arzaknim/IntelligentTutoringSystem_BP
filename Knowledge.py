import pandas as pd
import random as rng


class Knowledge:

    def __init__(self):
        self.elements = pd.read_csv('periodic_table.csv').iloc[:, 1]
        self.knowledge_dct = {}
        self.set_random_knowledge()
        s_block = []
        self.halogens_dct = self.get_dct_of_block(s_block)

    def set_random_knowledge(self):
        for element in self.elements:
            self.knowledge_dct[element] = rng.randint(0, 1)

    def get_dct_of_block(self, block_list: list[str]) -> dict:
        return dict((element, self.knowledge_dct[element]) for element in block_list if element in self.knowledge_dct)




