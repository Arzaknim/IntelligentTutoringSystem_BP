from BlockEnviroment import BlockEnvironment
from PeriodicTable import PeriodicTable
from Student import Student

if __name__ == '__main__':
    pt = PeriodicTable()
    student = Student(0.6, 0.2, 3, 1)
    s_env = BlockEnvironment('S', pt.get_s_block_dct(), student)
