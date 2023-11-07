class Student:

    # basically a config
    def __init__(self, lr, fr, goal_mark, block_strength):
        self.learning_rate = lr
        self.forgetting_rate = fr
        self.goal_mark = goal_mark
        self.block_strength = block_strength
