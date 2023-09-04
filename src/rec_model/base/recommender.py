from metric.rating import 
from metric.ranking import
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

    def predict_rank(self, u):
        pass
    
    def check_rating_boundary(self,prediction):
        
        
    def eval_rating(self):
        


    def eval_ranking(self):
        pass