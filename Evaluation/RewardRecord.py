class RewardRecord:
    def __init__(self, path):
        self.path = path

    def record(self, reward):
        with open(self.path, 'a') as f:
            f.write(str(reward)+'\n')