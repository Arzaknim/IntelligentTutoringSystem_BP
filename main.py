from PeriodicTable import PeriodicTable
from MasterAgent import MasterAgent

if __name__ == '__main__':
    # df = pd.read_csv('periodic_table.csv')
    # df = df.iloc[:, [1, 2]]
    # print(df)
    # print(kn.elements_dict)
    # print(kn.knowledge_dct)
    pt = PeriodicTable()
    ma = MasterAgent(pt)
