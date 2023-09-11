import tensorflow as tf

def bpr_loss(pos_scores, neg_scores):
    """
    Args:
        :param pos_scores: A tensor with shape of [batch_size, 1].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
    :return:
    """
    loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores)))
    return loss
