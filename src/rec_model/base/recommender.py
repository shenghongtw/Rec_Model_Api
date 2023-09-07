from metric.rating import rating_measure
from metric.ranking import ranking_measure
class Recommeder:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set

    def init_model(self):
        pass

    def save(self):
        pass

    def train(self):
        pass

    def predict_rate(self, u, i):
        pass

    def predict_score(self, u, i):
        pass

    def check_rating_boundary(self,prediction):
        if prediction > self.data.rScale[-1]:
            return self.data.rScale[-1]
        elif prediction < self.data.rScale[0]:
            return self.data.rScale[0]
        else:
            return round(prediction,3)
        
    def eval_rating(self):
        pass

    def eval_ranking(self):
        pass