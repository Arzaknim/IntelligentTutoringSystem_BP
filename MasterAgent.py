from PeriodicTable import PeriodicTable
from SlaveAgent import SlaveAgent


class MasterAgent:

    def __init__(self, pt: PeriodicTable):
        self.s_agent = SlaveAgent(pt.get_s_block_dct())
        self.p_agent = SlaveAgent(pt.get_p_block_dct())
        self.d_agent = SlaveAgent(pt.get_d_block_dct())
        self.f_agent = SlaveAgent(pt.get_f_block_dct())

    # actions
    def teach_sblock(self):
        print('You should study the S Block of the periodic table')
        self.s_agent.test()

    def teach_pblock(self):
        print('You should study the P Block of the periodic table')
        self.p_agent.test()

    def teach_dblock(self):
        print('You should study the D Block of the periodic table')
        self.d_agent.test()

    def teach_fblock(self):
        print('You should study the F Block of the periodic table')
        self.f_agent.test()

    def assessment(self):
        # a big test, which checks the student's knowledge from all 4 blocks
        pass