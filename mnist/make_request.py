import json
import random
from time import time

import numpy as np
import requests
from matplotlib import pyplot as plt
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def show(idx, title):
    plt.figure()
    plt.imshow(test_images[idx].reshape(28, 28))
    plt.axis('off')
    plt.title('\n\n{}'.format(title), fontdict={'size': 16})
    plt.show()


rando = random.randint(0, len(test_images) - 1)
# show(rando, 'An Example Image: {}'.format(class_names[test_labels[rando]]))


data = json.dumps({"signature_name": "serving_default", "instances": [test_images[rando].tolist()]})
# print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))


headers = {"content-type": "application/json"}
start = time()
json_response = requests.post('http://localhost:8502/v1/models/fashion_model/versions/1:predict', data=data,
                              headers=headers)
elapsed = (time() - start)
print(json_response.text)
predictions = json.loads(json_response.text)['predictions']

show(rando, 'predict: {} , actually: {} '.format(
    np.argmax(predictions[0]), test_labels[rando]))

print('predict: {} , actually: {}'.format(
    np.argmax(predictions[0]), test_labels[rando]),
    ", Time used:{0}ms".format(round(elapsed * 1000, 2)))
