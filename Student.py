class Student:

    # basically a config
    def __init__(self, lr, fr, starting_mark, goal_mark):
        self.learning_rate = lr
        self.forgetting_rate = fr
        self.starting_mark = starting_mark
        self.goal_mark = goal_mark
