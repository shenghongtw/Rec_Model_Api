import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import tensorflow as tf
from typing import Any
from rec_model.model import BPR

class Model:
    model_path: str
    model: Any

    def __init__(self, model_path):
        print("Load Basic Service")
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path, custom_objects={'BPR': BPR})

    def recommend_top_k(self, user_idx, items_idx=None, k=10):  
        user_idx = int(user_idx)
        if items_idx:
            items_idx = list(map(int, items_idx))
            scores = self.model.predict_score(user_idx, items_idx)
            _, indices = tf.math.top_k(scores, k=k)
            topk_indices = [items_idx[idx] for idx in indices.numpy()]
        else:
            items_idx = tf.range(self.model.item_embedding.input_dim).numpy()
            scores = self.model.predict_score(user_idx, items_idx)
            _, indices = tf.math.top_k(scores, k=k)
            topk_indices = [items_idx[idx] for idx in indices.numpy()]
        # 因為int32不是json serializable，所以要轉成int
        topk_indices = list(map(int, topk_indices))
        return topk_indices
    