# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
from tensorflow_estimator import estimator

COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
BATCH_SIZE = 100
STEPS = 10000

# load data
y_name = 'Species'

train = pd.read_csv('data/iris_training.csv', names=COLUMN_NAMES, header=0)
train_x, train_y = train, train.pop(y_name)

test = pd.read_csv('data/iris_test.csv', names=COLUMN_NAMES, header=0)
test_x, test_y = test, test.pop(y_name)


# prepare input / eval fn
def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    inputs = (features, labels) if labels is not None else features
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset

hook = estimator.ProfilerHook(save_steps=300, output_dir='./time/', show_memory=True, show_dataflow=True)
feature_columns = [tf.feature_column.numeric_column(key=key)
                   for key in train_x.keys()]

test = tf.feature_column.numeric_column(train_x.keys()[0], default_value=0.0)
test = tf.feature_column.bucketized_column(test,[0.1,1,100])
test_emb = tf.feature_column.embedding_column(test, 10)
feature_columns.append(test_emb)

session_config = tf.ConfigProto()

mirrored_strategy = tf.distribute.MirroredStrategy()


config = estimator.RunConfig(
    train_distribute=mirrored_strategy,
    eval_distribute=mirrored_strategy,
)

classifier = estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    config=config)

classifier.train(
    input_fn=lambda: train_input_fn(train_x, train_y, batch_size=BATCH_SIZE),hooks=[hook],
    steps=STEPS)

# evaluate
eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_x, test_y, batch_size=BATCH_SIZE))

print('Test set accuracy: {accuracy:0.3f}'.format(**eval_result))

# predict
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

predictions = classifier.predict(
    input_fn=lambda: eval_input_fn(predict_x, labels=None, batch_size=BATCH_SIZE))

for prediction, expect in zip(predictions, expected):
    class_id = prediction['class_ids'][0]
    probability = prediction['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        SPECIES[class_id], 100 * probability, expect))

# export model

from tensorflow import FixedLenFeature

feature_specification = {
    'SepalLength': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=None),
    'SepalWidth': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=None),
    'PetalLength': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=None),
    'PetalWidth': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=None)
}
# feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)

features = {
    'SepalLength': tf.placeholder(dtype=tf.float32, shape=(1), name='SepalLength'),
    'SepalWidth': tf.placeholder(dtype=tf.float32, shape=(1), name='SepalWidth'),
    'PetalLength': tf.placeholder(dtype=tf.float32, shape=(1), name='PetalLength'),
    'PetalWidth': tf.placeholder(dtype=tf.float32, shape=(1), name='PetalWidth')
}

# Can pass the key-value format to http REST api directly.
# curl -d '{"signature_name": "predict","instances": [{"SepalLength":[5.1],"SepalWidth":[3.3],"PetalLength":[1.7],"PetalWidth":[0.5]}]}' -X POST http://localhost:8501/v1/models/iris:predict

# If using shape=(None,1), the input shape should be (?,1). Means two dimensions
# saved_model_cli run --dir /home/wangrc/github/summaries/serving/estimator/export_raw/1560322385 \
#   --tag_set serve --signature_def predict \
#   --input_exprs 'SepalLength=[[5.1],[5.1]];SepalWidth=[[3.3],[3]];PetalLength=[[1.7],[3]];PetalWidth=[[0.5],[2]]'

serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)
export_dir = classifier.export_savedmodel('export_raw', serving_input_receiver_fn)

# Only works in this way
# If using http REST api, the json can not parse the tf.Example string.
# saved_model_cli run --dir /home/wangrc/Downloads/tf-serve-master/export_parsing/1560302102 \
#   --tag_set serve --signature_def predict \
#   --input_examples 'examples=[{"SepalLength":[5.1],"SepalWidth":[3.3],"PetalLength":[1.7],"PetalWidth":[0.5]}]'

# serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_specification)
# export_dir = classifier.export_savedmodel('export_parsing', serving_input_receiver_fn)


print('Exported to {}'.format(export_dir))
