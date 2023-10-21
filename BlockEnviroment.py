import gymnasium as gym
import numpy as np
import random


class BlockEnvironment(gym.Env):
    def __init__(self, name, block_dict, student):
        self.name = name
        self.student = student
        # Actions we can take are show materials, test, quit
        self.action_space = gym.spaces.Discrete(3)
        # Grade 1 to 5 (1 is best performance, 5 is worst performance)
        self.observation_space = gym.spaces.Discrete(5, start=1)
        self.gt = block_dict
        self.knowledge_space = {}
        self.set_graded_knowledge(student.starting_mark)
        self.state = self.knowledge2observational()
        self.study_length = 60

    def step(self, action):
        reward = 0
        # shows materials
        if action == 0:
            self.learn()
        # tests student
        elif action == 1:
            self.test()
        elif action == 2:
            done = True
        old_state = self.state
        self.state = self.knowledge2observational()
        # Reduce shower length by 1 second
        self.study_length -= 1

        # Calculate reward
        diff = old_state - self.state
        if diff > 1:
            reward += 20 * diff
        else:
            reward -= 20 * diff

        # Check if study_session is done
        if self.study_length <= 0:
            done = True
        else:
            done = False

        info = {}

        # Return step information
        return self.state, reward, done, info

    def reset(self, seed=None, options=None):
        self.set_graded_knowledge(self.student.starting_mark)
        self.state = self.knowledge2observational()
        self.study_length = 60
        return self.state

    def render(self):
        pass

    def set_random_knowledge(self):
        result = {}
        for symbol in self.gt.keys():
            result[symbol] = random.randint(0, 1)

        self.knowledge_space = result

    def set_graded_knowledge(self, grade=1):
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

        knowledge_grade = 0
        while knowledge_grade != grade:
            result = {}
            for symbol in self.gt.keys():
                if random.random() < correct_rate:
                    result[symbol] = 1
                else:
                    result[symbol] = 0

            self.knowledge_space = result
            knowledge_grade = self.knowledge2observational()

    def knowledge2observational(self):
        n_symbols = len(self.knowledge_space.keys())
        n_correct = 0
        for symbol in self.knowledge_space.keys():
            if self.knowledge_space[symbol] == 1:
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

    def learn(self, sim=True):
        if sim:
            for symbol in self.knowledge_space.keys():
                rn = random.random()
                if self.knowledge_space[symbol] == 1:
                    if rn < self.student.forgetting_rate:
                        self.knowledge_space[symbol] = 0
                else:
                    if rn > self.student.learning_rate:
                        self.knowledge_space[symbol] = 1

        print(f'You should study the {self.name} block of the periodic table')

    def test(self, sim=True):
        lst = []
        for symbol in self.knowledge_space.keys():
            if self.knowledge_space[symbol] == 0:
                lst = lst + [self.knowledge_space[symbol]]

        lst = random.choices(lst, k=3)
        for i in range(3):
            print(f"You should check the element {lst[i]}.")

        if sim:
            for symbol in lst:
                if random.random() < self.student.learning_rate:
                    self.knowledge_space[symbol] = 1
                else:
                    self.knowledge_space[symbol] = 0

