import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time import time
from tensorflow.keras.optimizers import Adam

from rec_model.model.bpr import BPR
from rec_model.data.datasets import movielens as ml
from rec_model.evaluate import eval_pos_neg



neg_num = 10
test_neg_num = 100
embed_dim = 64
use_l2norm = False
embed_reg = 0.0
learning_rate = 0.01
epochs = 2
batch_size = 64
k = 10
file_path = 'test/data/ml-1m/ratings.dat'
def main():
    # TODO: 1. Split Data
    train_path, val_path, test_path, meta_path = ml.split_data(file_path=file_path)
    # else:
    #     train_path, val_path, test_path, meta_path = train_path, val_path, test_path, meta_path
    with open(meta_path) as f:
        max_user_num, max_item_num = [int(x) for x in f.readline().strip('\n').split('\t')]
    # TODO: 2. Load Data
    train_data = ml.load_data(train_path, neg_num, max_item_num)
    val_data = ml.load_data(val_path, neg_num, max_item_num)
    test_data = ml.load_data(test_path, test_neg_num, max_item_num)
    # TODO: 3. Set Model Hyper Parameters.
    model_params = {
        'user_num': max_user_num + 1,
        'item_num': max_item_num + 1,
        'embed_dim': embed_dim,
        'use_l2norm': use_l2norm,
        'embed_reg': embed_reg
    }
    # TODO: 4. Build Model
    model = BPR(**model_params)
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    # TODO: 5. Fit Model
    for epoch in range(1, epochs + 1):
        t1 = time()
        model.fit(
            x=train_data,
            epochs=1,
            validation_data=val_data,
            batch_size=batch_size
        )
        t2 = time()
        eval_dict = eval_pos_neg(model, test_data, ['hr', 'mrr', 'ndcg'], k, batch_size)
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, MRR = %.4f, NDCG = %.4f, '
              % (epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['mrr'], eval_dict['ndcg']))


if __name__ == '__main__':
    main()