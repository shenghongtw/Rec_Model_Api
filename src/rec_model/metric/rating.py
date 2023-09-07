import math
class rating_measure:
    def __init__(self):
        pass
    
    @staticmethod
    def rmse(self, prediction, ground_truth):
        return math.sqrt(sum([(prediction[i] - ground_truth[i]) ** 2 for i in range(len(prediction))]) / len(prediction))