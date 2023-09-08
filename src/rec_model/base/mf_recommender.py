import numpy as np
import tensorflow.compat.v1 as tf

from util import find_topk_items
from base.recommender import Recommeder
from metric.rating import rating_measure
from metric.ranking import ranking_measure

class MfRecommender(Recommeder):
    def __init__(self, train_set, test_set,
                 emb_size=10, max_epoch=100, learning_rate=0.01,
                 batch_size=100, reg_u=0.01, reg_i=0.01, reg_bias=0.01):
        super().__init__(train_set, test_set)
        self.emb_size = emb_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.reg_bias = reg_bias
        self.best_performance = []

    def init_model(self):
        self.u_idx = tf.placeholder(tf.int32, [None], name='u_idx')
        self.i_idx = tf.placeholder(tf.int32, [None], name='i_idx')
        self.r_label = tf.placeholder(tf.float32, [None], name='rating')

        self.U = tf.Variable(tf.truncated_normal([self.train_set.num_user, self.emb_size], stddev=0.01), name='user')
        self.I = tf.Variable(tf.truncated_normal([self.train_set.num_item, self.emb_size], stddev=0.01), name='item')
        self.user_biases = tf.Variable(tf.truncated_normal(shape=[self.num_users, 1], stddev=0.005), name='U')
        self.item_biases = tf.Variable(tf.truncated_normal(shape=[self.num_items, 1], stddev=0.005), name='U')

        self.user_bias = tf.nn.embedding_lookup(self.user_biases, self.u_idx)
        self.item_bias = tf.nn.embedding_lookup(self.item_biases, self.v_idx)
        self.user_embedding = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.item_embedding = tf.nn.embedding_lookup(self.V, self.v_idx)

    def predict_rate(self, u, i):
        if self.data.containsUser(u) and self.data.containsItem(i):
            return self.P[self.data.user[u]].dot(self.Q[self.data.item[i]])
        elif self.data.containsUser(u) and not self.data.containsItem(i):
            return self.data.userMeans[u]
        elif not self.data.containsUser(u) and self.data.containsItem(i):
            return self.data.itemMeans[i]
        else:
            return self.data.globalMean
    
    def predict_score(self, u):
        if self.data.contains_user(u):
            return self.P[self.data.user[u]].dot(self.Q.T)
        else:
            return np.array([self.data.globalMean] * self.data.num_item)
        
    def rating_performance(self):
        ground_truth = []
        prediction_list = []
        for data in self.data.testData:
            user, item, rating = data
            # predict
            prediction = self.predict_rate(user, item)
            pred = self.check_rating_boundary(prediction)
            ground_truth.append(rating)
            prediction_list.append(pred)
        self.performance = rating_measure.rmse(prediction_list, ground_truth)
        return self.performance

    def ranking_performance(self, topk):
        rec_list = {}
        for user in self.data.testSet_u:
            score_list = self.predict_score(user)
            topk_item_ids = find_topk_items(topk, score_list)
            rec_list[user] = topk_item_ids
        hits = ranking_measure.hits(self.data.testSet_u, rec_list)
        precision = ranking_measure.precision(hits, topk)
        recall = ranking_measure.recall(hits, data)
        self.performance = (precision, recall)
        return self.performance
        
        