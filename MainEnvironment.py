import gymnasium as gym
from gymnasium.spaces import Discrete
from BlockEnviroment import BlockEnvironment


class MainEnvironment(gym.Env):

    def __init__(self, pt, student):
        self.pt = pt
        self.student = student
        self.s_block_env = BlockEnvironment('S', self.pt.get_s_block_dct(),
                                            self.student, self.student.block_strength[0])
        self.p_block_env = BlockEnvironment('P', self.pt.get_p_block_dct(),
                                            self.student, self.student.block_strength[1])
        self.d_block_env = BlockEnvironment('D', self.pt.get_d_block_dct(),
                                            self.student, self.student.block_strength[2])
        self.f_block_env = BlockEnvironment('F', self.pt.get_f_block_dct(),
                                            self.student, self.student.block_strength[3])
        self.action_space = Discrete(6)
        self.observation_space = Discrete(5)
        self.state = self.knowledge2observational()
        self.time_step = 60

    def step(self, action):
        block_env_action = self.s_block_env.action_space.sample()
        reward = 0
        info = ""
        done = False
        if action == 0:
            n_state, reward, done, info = self.s_block_env.step(block_env_action)
        elif action == 1:
            n_state, reward, done, info = self.p_block_env.step(block_env_action)
        elif action == 2:
            n_state, reward, done, info = self.d_block_env.step(block_env_action)
        elif action == 3:
            n_state, reward, done, info = self.f_block_env.step(block_env_action)
        elif action == 4:
            old_state = self.state
            self.state, assessment_reward = self.assessment()
            reward += assessment_reward
            if self.state > old_state:
                reward += (self.state - old_state) * 50
            elif self.state < old_state:
                reward -= (old_state - self.state) * 50
        elif action == 5:
            done = True

        if self.time_step <= 0 or done:
            done = True
        else:
            done = False

        return self.state, reward, done, info

    def reset(self, seed=None, options=None):
        self.s_block_env = BlockEnvironment('S', self.pt.get_s_block_dct(),
                                            self.student, self.student.block_strength[0])
        self.p_block_env = BlockEnvironment('P', self.pt.get_p_block_dct(),
                                            self.student, self.student.block_strength[1])
        self.d_block_env = BlockEnvironment('D', self.pt.get_d_block_dct(),
                                            self.student, self.student.block_strength[2])
        self.f_block_env = BlockEnvironment('F', self.pt.get_f_block_dct(),
                                            self.student, self.student.block_strength[3])
        self.time_step = 20

    def assessment(self):
        reward = 0
        print("main environment assessment")
        reward += self.s_block_env.learn(True)
        reward += self.p_block_env.learn(True)
        reward += self.d_block_env.learn(True)
        reward += self.f_block_env.learn(True)
        # s_knowledge = self.s_block_env.knowledge_space
        # p_knowledge = self.p_block_env.knowledge_space
        # d_knowledge = self.d_block_env.knowledge_space
        # f_knowledge = self.f_block_env.knowledge_space
        # result_dict = dict(s_knowledge)
        # result_dict.update(p_knowledge)
        # result_dict.update(d_knowledge)
        # result_dict.update(f_knowledge)
        return self.knowledge2observational(), reward

    def knowledge2observational(self):
        s_grade = self.s_block_env.knowledge2observational()
        p_grade = self.p_block_env.knowledge2observational()
        d_grade = self.d_block_env.knowledge2observational()
        f_grade = self.f_block_env.knowledge2observational()
        result = s_grade + p_grade + d_grade + f_grade
        return round(result)

    def render(self):
        pass
