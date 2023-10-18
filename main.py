import pandas as pd

from Knowledge import Knowledge

if __name__ == '__main__':
    # df = pd.read_csv('periodic_table.csv')
    # df = df.iloc[:, [1, 2]]
    # print(df)
    kn = Knowledge()
    print(kn.elements_dict)
