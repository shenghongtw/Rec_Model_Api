import math

class ranking_measure:
    def __init__(self):
        pass
    
    @staticmethod
    def hits(self, prediction, origin):
        hitCount = {}
        for user in origin:
            items = list(origin[user].keys())
            prediction_list = [item[0] for item in prediction[user]]
            hitCount[user] = len(set(items).intersection(set(prediction_list)))
        return hitCount

    @staticmethod
    def recall(self, hits, origin):
        recallList = [hits[user]/len(origin[user]) for user in hits]
        recall = sum(recallList) / len(recallList)
        return recall
    
    @staticmethod
    def precision(self, hits, k):
        precisionList = [hits[user]/k for user in hits]
        precision = sum(precisionList) / len(precisionList)
        return precision
    
    @staticmethod
    def auc(self, prediction, origin):
        aucList = []
        for user in origin:
            items = list(origin[user].keys())
            prediction_list = [item[0] for item in prediction[user]]
            aucList.append(self.auc_single_user(items, prediction_list))
        auc = sum(aucList) / len(aucList)
        return auc