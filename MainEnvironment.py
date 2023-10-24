import gymnasium as gym
from gymnasium.spaces import Discrete
from BlockEnviroment import BlockEnvironment


class MainEnvironment(gym.Env):

    def __init__(self, pt, student):
        self.pt = pt
        self.student = student
        self.s_block_env = BlockEnvironment('S', pt.get_s_block_dct(), student)
        self.p_block_env = BlockEnvironment('P', pt.get_p_block_dct(), student)
        self.d_block_env = BlockEnvironment('D', pt.get_d_block_dct(), student)
        self.f_block_env = BlockEnvironment('F', pt.get_f_block_dct(), student)
        self.action_space = Discrete(6)
        self.observation_space = Discrete(5)
        self.state = self.knowledge2observational()
        self.time_step = 60
        self.done = False

    def step(self, action):
        action = self.s_block_env.action_space.sample()
        reward = 0
        if action == 0:
            n_state, reward, done, info = self.s_block_env.step(action)
        elif action == 1:
            n_state, reward, done, info = self.p_block_env.step(action)
        elif action == 2:
            n_state, reward, done, info = self.d_block_env.step(action)
        elif action == 3:
            n_state, reward, done, info = self.f_block_env.step(action)
        elif action == 4:
            self.state = self.assessment()
        elif action == 5:
            self.done = True

        return self.state, reward, done, info

    def reset(self, seed=None, options=None):
        self.s_block_env = BlockEnvironment('S', self.pt.get_s_block_dct(), self.student)
        self.p_block_env = BlockEnvironment('P', self.pt.get_p_block_dct(), self.student)
        self.d_block_env = BlockEnvironment('D', self.pt.get_d_block_dct(), self.student)
        self.f_block_env = BlockEnvironment('F', self.pt.get_f_block_dct(), self.student)
        self.time_step = 20
        self.done = False

    def assessment(self):
        s_knowledge = self.s_block_env.knowledge_space
        p_knowledge = self.p_block_env.knowledge_space
        d_knowledge = self.d_block_env.knowledge_space
        f_knowledge = self.f_block_env.knowledge_space
        result_dict = dict(s_knowledge)
        result_dict.update(p_knowledge)
        result_dict.update(d_knowledge)
        result_dict.update(f_knowledge)
        # TODO add questions/simulation of learning
        return 1

    def knowledge2observational(self):
        s_grade = self.s_block_env.knowledge2observational()
        p_grade = self.p_block_env.knowledge2observational()
        d_grade = self.d_block_env.knowledge2observational()
        f_grade = self.f_block_env.knowledge2observational()
        result = s_grade + p_grade + d_grade + f_grade
        return round(result)

    def render(self):
        pass
