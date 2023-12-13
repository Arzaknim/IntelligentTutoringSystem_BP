import random

from IntelligentTutoringSystem_BP.Question import Question


def set_block_knowledge(gt):
    result = {}
    for symbol in gt.keys():
        result[symbol] = Question()

    knowledge_space = result

    return knowledge_space


def block_knowledge2grade(block_knowledge_space):
    n_symbols = len(block_knowledge_space.keys())
    n_correct = 0
    for symbol in block_knowledge_space.keys():
        if block_knowledge_space[symbol].get_last_answer() == 1:
            n_correct += 1

    percents = n_correct / n_symbols * 100

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


def forget(block_knowledge_space, fr):
    reward = 0
    for symbol in block_knowledge_space.keys():
        rn = random.random()
        if block_knowledge_space[symbol].get_last_answer() == 1:
            # 2x*(y**2)
            if rn < fr * (2 * block_knowledge_space[symbol].times_asked /
                          block_knowledge_space[symbol].times_correct ** 2):
                block_knowledge_space[symbol].last_answer = 0
                reward -= 2

    return reward


def grade2onehot(grade):
    return [1 if x + 1 == grade else 0 for x in range(5)]
