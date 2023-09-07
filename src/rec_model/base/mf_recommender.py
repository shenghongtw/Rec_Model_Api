import numpy as np
from random import shuffle
import tensorflow.compat.v1 as tf
from base.recommender import Recommeder

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
        if self.data.containsUser(u):
            return self.P[self.data.user[u]].dot(self.Q.T)
        else:
            return np.array([self.data.globalMean] * self.data.num_item)