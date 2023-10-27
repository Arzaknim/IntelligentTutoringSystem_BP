class Question:

    def __init__(self, last_answer, times_asked, times_correct):
        self.last_answer = last_answer
        self.times_asked = times_asked
        self.times_correct = times_correct

    def update(self, answer):
        self.last_answer = answer
        self.times_asked += 1
        if answer == 1:
            self.times_correct += 1

    def get_last_answer(self):
        return self.last_answer

    def get_times_asked(self):
        return self.times_asked

    def get_times_correct(self):
        return self.times_correct
