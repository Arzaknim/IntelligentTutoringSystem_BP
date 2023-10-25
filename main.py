from MainEnvironment import MainEnvironment
from PeriodicTable import PeriodicTable
from Student import Student
import operator

if __name__ == '__main__':
    pt = PeriodicTable()
    block_strength = [1, 1, 1, 1]
    student = Student(0.6, 0.2, 3, 1, block_strength)
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
