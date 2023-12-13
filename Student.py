class Student:

    # basically a config
    def __init__(self, lr, fr, goal_mark, block_strength):
        self.learning_rate = lr
        self.forgetting_rate = fr
        self.goal_mark = [1 if x+1 == goal_mark else 0 for x in range(5)]
        self.block_strength = block_strength
