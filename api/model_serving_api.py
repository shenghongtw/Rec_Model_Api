import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import uvicorn
import json
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

from api.model import Model
from rec_model.model import BPR

model_name = 'BPR'

model_path = f'save_model/{model_name}'

app = FastAPI() # initiate API
model = Model(model_path)
tf.keras.models.load_model(model_path, custom_objects={f'{model_name}': BPR})

class ModelParams(BaseModel):
    user: str

@app.post("/submit")
def submit(params: ModelParams):
    topk_item = model.recommend_top_k(params.user)
    topk_item = json.dumps(topk_item)
    return {"topk_item": topk_item}

if __name__ == "__main__": 
     uvicorn.run(app, host="0.0.0.0", port=8000)