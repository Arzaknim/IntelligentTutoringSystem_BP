import pandas as pd
import random as rng


class Knowledge:

    def __init__(self):
        self.elements_df = pd.read_csv('periodic_table.csv').iloc[:, [1, 2]].to_dict()
        self.elements_dict = self.elements_df2dict()
        self.knowledge_dct = {}
        self.set_random_knowledge()
        s_block_list = [1, 2, 3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88]
        p_block_list = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 31, 32, 33, 34, 35, 36, 49,
                        50, 51, 52, 53, 54, 81, 82, 83, 84, 85, 86, 113, 114, 115, 116, 117, 118]
        d_block_list = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 57,
                        72, 73, 74, 75, 76, 77, 78, 79, 80, 89, 104, 105, 106, 107, 108, 109, 110, 111, 112]
        f_block_list = [x for x in range(58, 72)] + [x for x in range(90, 104)]
        self.s_block_dct = self.get_dct_of_block(s_block_list)
        self.p_block_dct = self.get_dct_of_block(p_block_list)
        self.d_block_dct = self.get_dct_of_block(d_block_list)
        self.f_block_dct = self.get_dct_of_block(f_block_list)

    def elements_df2dict(self):
        result = {}
        length = len(self.elements_df['Element'])
        elements = self.elements_df['Element']
        symbols = self.elements_df['Symbol']
        for idx in range(length):
            symbol = symbols[idx]
            element = elements[idx]
            result[symbol] = element

        return result

    def set_random_knowledge(self):
        for symbol in self.elements_dict.keys():
            self.knowledge_dct[symbol] = rng.randint(0, 1)

    def get_dct_of_block(self, block_list: list[int]) -> dict:
        result = {}
        knowledge_list = list(self.knowledge_dct.items())
        for idx in block_list:
            symbol, answer_bool = knowledge_list[idx - 1]
            result[symbol] = answer_bool

        return result





