##############################
#
# call tfserving using REST API demo
#
##############################

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from time import time

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


def show(idx, title):
    plt.figure()
    plt.imshow(test_images[idx].reshape(28, 28))
    plt.axis('off')
    plt.title('\n\n{}'.format(title), fontdict={'size': 16})
    plt.show()


import random
import requests
import json

correct=0
error=0
def test_post():
    acc=0
    rando = random.randint(0, len(test_images) - 1)
    data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:400].tolist()})
    # print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))
    headers = {"content-type": "application/json"}
    start = time()
    json_response = requests.post('http://algorithmsdemo.2345.cn/v1/models/mnist/versions/1:predict', data=data, headers=headers)
    elapsed = (time() - start)
    # print(json_response.text)
    # predictions = json.loads(json_response.text)['predictions']

    # print("Time used:{0}ms".format(round(elapsed * 1000, 2)))
    # print('predict: {} , actually: {} '.format(
    #     np.argmax(predictions[0]), test_labels[rando]))
    # show(rando, 'predict: {} , actually: {} '.format(
    #   np.argmax(predictions[0]), test_labels[rando]))
    return elapsed

sec=[]
req_count=100
for i in range(req_count):
    sec.append(test_post())
print('average cost time {0} ms'.format(np.sum(sec)/req_count*1000))