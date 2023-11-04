import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
import random

from Question import Question


class MainEnvironment(gym.Env):

    def __init__(self, pt, student):
        self.sim = True
        self.pt = pt
        self.student = student
        self.s_block_knowledge = self.set_block_knowledge(student.starting_mark, self.pt.get_s_block_dct())
        self.p_block_knowledge = self.set_block_knowledge(student.starting_mark, self.pt.get_p_block_dct())
        self.d_block_knowledge = self.set_block_knowledge(student.starting_mark, self.pt.get_d_block_dct())
        self.f_block_knowledge = self.set_block_knowledge(student.starting_mark, self.pt.get_f_block_dct())
        # study each block (4), test each block (4), assessment = 9
        self.action_space = Discrete(5)
        self.observation_space = MultiDiscrete([5, 5, 5, 5])
        self.state = self.knowledge2observational()
        self.last_grade = self.knowledge2grade()
        self.time_step = 10

    def step(self, action):
        reward = 0
        info = ""
        done = False
        if action == 0:
            reward = self.learn('s')
        elif action == 1:
            reward = self.learn('p')
        elif action == 2:
            reward = self.learn('d')
        elif action == 3:
            reward = self.learn('f')
        elif action == 4:
            reward = self.assessment()

        self.time_step -= 1
        if self.time_step <= 0 or done:
            done = True
        else:
            done = False

        return self.state, reward, done, info

    def reset(self, seed=None, options=None):
        self.s_block_knowledge = self.set_block_knowledge(self.student.starting_mark, self.pt.get_s_block_dct())
        self.p_block_knowledge = self.set_block_knowledge(self.student.starting_mark, self.pt.get_p_block_dct())
        self.d_block_knowledge = self.set_block_knowledge(self.student.starting_mark, self.pt.get_d_block_dct())
        self.f_block_knowledge = self.set_block_knowledge(self.student.starting_mark, self.pt.get_f_block_dct())
        self.time_step = 10

    def learn_block(self, name):
        reward = 0
        reward += self.learn(name)
        knowledge_space = None
        idx = None
        if name == 's':
            knowledge_space = self.s_block_knowledge
            idx = 0
        elif name == 'p':
            knowledge_space = self.p_block_knowledge
            idx = 1
        elif name == 'd':
            knowledge_space = self.d_block_knowledge
            idx = 2
        elif name == 'f':
            knowledge_space = self.f_block_knowledge
            idx = 3

        old_grade = self.state[idx]
        new_grade = self.block_knowledge2observational(knowledge_space)
        reward += (old_grade - new_grade) * 8

        self.state[idx] = new_grade
        info = {}

        # Return step information
        return reward

    def learn(self, name):
        reward = 0
        lr = None
        knowledge_space = None
        if name == 's':
            lr = self.student.learning_rate * self.student.block_strength[0]
            knowledge_space = self.s_block_knowledge
        elif name == 'p':
            lr = self.student.learning_rate * self.student.block_strength[1]
            knowledge_space = self.p_block_knowledge
        elif name == 'd':
            lr = self.student.learning_rate * self.student.block_strength[2]
            knowledge_space = self.d_block_knowledge
        elif name == 'f':
            lr = self.student.learning_rate * self.student.block_strength[3]
            knowledge_space = self.f_block_knowledge
        if self.sim:
            for symbol in knowledge_space.keys():
                rn = random.random()
                if knowledge_space[symbol].get_last_answer() == 0:
                    if rn < lr:
                        knowledge_space[symbol].update(1)
                        reward += 3
        print(f'Learning session of the {name} block')
        reward += self.forget(self.s_block_knowledge)
        reward += self.forget(self.p_block_knowledge)
        reward += self.forget(self.d_block_knowledge)
        reward += self.forget(self.f_block_knowledge)

        return reward

    def assessment(self):
        reward = 0
        print("main environment assessment, (testing real knowledge)")
        assessment_grade = self.knowledge2grade()
        reward += (self.last_grade - assessment_grade) * 50
        self.last_grade = assessment_grade
        return reward

    def knowledge2grade(self):
        s_grade = self.block_knowledge2observational(self.s_block_knowledge)
        p_grade = self.block_knowledge2observational(self.p_block_knowledge)
        d_grade = self.block_knowledge2observational(self.d_block_knowledge)
        f_grade = self.block_knowledge2observational(self.f_block_knowledge)
        result = (s_grade + p_grade + d_grade + f_grade) / 4
        return round(result)

    def knowledge2observational(self):
        s_grade = self.block_knowledge2observational(self.s_block_knowledge)
        p_grade = self.block_knowledge2observational(self.p_block_knowledge)
        d_grade = self.block_knowledge2observational(self.d_block_knowledge)
        f_grade = self.block_knowledge2observational(self.f_block_knowledge)
        return [s_grade, p_grade, d_grade, f_grade]

    def render(self):
        pass

    def set_block_knowledge(self, grade, gt):
        if grade == 1:
            correct_rate = 0.95
        elif grade == 2:
            correct_rate = 0.85
        elif grade == 3:
            correct_rate = 0.65
        elif grade == 4:
            correct_rate = 0.375
        else:
            correct_rate = 0.125

        knowledge_space = None
        knowledge_grade = 0
        while knowledge_grade != grade:
            result = {}
            for symbol in gt.keys():
                if random.random() < correct_rate:
                    result[symbol] = Question(1, 1, 1)
                else:
                    result[symbol] = Question(0, 1, 0)

            knowledge_space = result
            knowledge_grade = self.block_knowledge2observational(result)

        return knowledge_space

    def block_knowledge2observational(self, block_knowledge_space):
        n_symbols = len(block_knowledge_space.keys())
        n_correct = 0
        for symbol in block_knowledge_space.keys():
            if block_knowledge_space[symbol].get_last_answer() == 1:
                n_correct += 1

        percents = n_correct/n_symbols*100

        if 100 >= percents >= 90:
            state = 1
        elif 89 >= percents >= 80:
            state = 2
        elif 79 >= percents >= 50:
            state = 3
        elif 49 >= percents >= 25:
            state = 4
        else:
            state = 5

        return state

    def forget(self, block_knowledge_space):
        reward = 0
        for symbol in block_knowledge_space.keys():
            rn = random.random()
            if block_knowledge_space[symbol].get_last_answer() == 1:
                if rn < self.student.forgetting_rate:
                    block_knowledge_space[symbol].update(0)
                    reward -= 3

        return reward
