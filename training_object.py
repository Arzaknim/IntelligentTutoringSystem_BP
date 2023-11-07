class TrainingObject:
    def __init__(self, memory, target_net, policy_net, optimizer):
        self.memory = memory
        self.target_net = target_net
        self.policy_net = policy_net
        self.optimizer = optimizer
        