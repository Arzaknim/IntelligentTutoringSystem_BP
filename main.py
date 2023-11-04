import torch

from MainEnvironment import MainEnvironment
from PeriodicTable import PeriodicTable
from Student import Student

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = PeriodicTable()
    block_strength = [1, 1, 0.2, 0.5]
    student = Student(0.7, 0.10, 4, 1, block_strength)
    env = MainEnvironment(pt, student)

    episodes = 100
    episodes_dict = {}
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        episodes_dict[episode] = score
        print('Episode:{} Score:{}'.format(episode, score))

    episodes_dict = {k: v for k, v in sorted(episodes_dict.items(), key=lambda item: item[1])}
    print(episodes_dict)
