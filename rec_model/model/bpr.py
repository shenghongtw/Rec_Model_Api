import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.regularizers import l2

from rec_model.util.loss import bpr_loss


class BPR(Model):
    def __init__(self, user_num, item_num, embed_dim, use_l2norm=False, embed_reg=0., seed=None):
        super(BPR, self).__init__()
        # user embedding
        self.user_embedding = Embedding(input_dim=user_num,
                                        input_length=1,
                                        output_dim=embed_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding = Embedding(input_dim=item_num,
                                        input_length=1,
                                        output_dim=embed_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # norm
        self.use_l2norm = use_l2norm
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs):
        # user info
        user_emb = self.user_embedding(tf.reshape(inputs['user'], [-1, ]))  # (None, embed_dim)
        # item info
        pos_item_emb = self.item_embedding(tf.reshape(inputs['pos_item'], [-1, ]))  # (None, embed_dim)
        neg_item_emb = self.item_embedding(inputs['neg_item'])  # (None, neg_num, embed_dim)
        # norm
        if self.use_l2norm:
            
            pos_item_emb = tf.math.l2_normalize(pos_item_emb, axis=-1)
            neg_item_emb = tf.math.l2_normalize(neg_item_emb, axis=-1)
            user_emb = tf.math.l2_normalize(user_emb, axis=-1)
        # calculate positive item scores and negative item scores
        pos_scores = tf.reduce_sum(tf.multiply(user_emb, pos_item_emb), axis=-1, keepdims=True)  # (None, 1)
        neg_scores = tf.reduce_sum(tf.multiply(tf.expand_dims(user_emb, axis=1), neg_item_emb), axis=-1)  # (None, neg_num)
        # add loss
        self.add_loss(bpr_loss(pos_scores, neg_scores)) # regularization loss是加在embedding嗎?
        # logits = tf.concat([pos_scores, neg_scores], axis=-1)
        # return logits
    
    def predict_score(self, user_idx, item_idx):
        score = tf.reduce_sum(tf.multiply(self.user_embedding(user_idx), self.item_embedding(item_idx)), axis=-1)
        return score

    def recommend_top_k(self, user_idx, items_idx, k=10):
        scores = self.predict_score(user_idx, items_idx)
        _, indices = tf.math.top_k(scores, k=k)
        return indices
        
    
    def get_user_vector(self, inputs):
        if len(inputs) < 2 and inputs.get('user') is not None:
            return self.user_embedding(inputs['user'])

    def summary(self):
        inputs = {
            'user': Input(shape=(), dtype=tf.int32),
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)  # suppose neg_num=1
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()