import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from time import time
from tensorflow.keras.optimizers import Adam

from rec_model.model.bpr import BPR
from rec_model.data.datasets import movielens as ml
from rec_model.evaluate import eval_pos_neg

def main():
    neg_num = 10
    test_neg_num = 100
    embed_dim = 64
    use_l2norm = False
    embed_reg = 0.0
    learning_rate = 0.01
    epochs = 1
    batch_size = 64
    k = 10
    file_path = "rec_model/test/data/ml-1m/ratings.dat"
    train_path = "rec_model/test/data/ml-1m/ml_train.txt"
    val_path = "rec_model/test/data/ml-1m/ml_val.txt"
    test_path = "rec_model/test/data/ml-1m/ml_test.txt"
    meta_path = "rec_model/test/data/ml-1m/ml_meta.txt"

    if os.path.exists(train_path):
        train_path, val_path, test_path, meta_path = train_path, val_path, test_path, meta_path
    else:
        train_path, val_path, test_path, meta_path = ml.split_data(file_path=file_path)
    with open(meta_path) as f:
        max_user_num, max_item_num = [int(x) for x in f.readline().strip('\n').split('\t')]

    train_data = ml.load_data(train_path, neg_num, max_item_num)
    val_data = ml.load_data(val_path, neg_num, max_item_num)
    test_data = ml.load_data(test_path, test_neg_num, max_item_num)

    model_params = {
        'user_num': max_user_num + 1,
        'item_num': max_item_num + 1,
        'embed_dim': embed_dim,
        'use_l2norm': use_l2norm,
        'embed_reg': embed_reg
    }

    model = BPR(**model_params)
    model.compile(optimizer=Adam(learning_rate=learning_rate))

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
        print(f'Iteration{epoch}, Fit [{t2 - t1}], Evaluate [{time() - t2}]: HR = {eval_dict["hr"]}, MRR = {eval_dict["mrr"]}, NDCG = {eval_dict["ndcg"]}')
        # TODO 依據performance來儲存模型
    model_name = type(model).__name__
    model.save(f'save_model/{model_name}')

if __name__ == '__main__':
    main()