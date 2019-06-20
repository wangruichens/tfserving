# REST remote call using estimator model with build_raw_serving_input_receiver_fn

import json
import requests
import numpy as np
from time import time

# tensorflow_model_server \
#   --rest_api_port=8501 \
#   --model_name=deepfm \
#   --model_base_path="/home/wangrc/Desktop/"

batching=[]
for i in range(1000):
    p = {
        "SepalLength": np.random.random(),
        "SepalWidth": np.random.random(),
        "PetalLength": np.random.random(),
        "PetalWidth": np.random.random()
    }
    batching.append(p)

data = json.dumps({"signature_name": "predict", "instances": batching})
# print(data)

headers = {"content-type": "application/json"}

start = time()
json_response = requests.post('http://ha05:8555/v1/models/iris:predict', data=data, headers=headers)
elapsed = (time() - start)
# print(json_response.text)
predictions = json.loads(json_response.text)['predictions']
print(predictions)
print("Time used:{0}ms".format(round(elapsed * 1000, 2)))
