from BlockEnviroment import BlockEnvironment
from PeriodicTable import PeriodicTable
from Student import Student

if __name__ == '__main__':
    pt = PeriodicTable()
    student = Student(0.6, 0.2, 3, 1)
    env = BlockEnvironment('S', pt.get_s_block_dct(), student)

    episodes = 100
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            # env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))
