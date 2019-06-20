# GRPC remote call using estimator model with build_raw_serving_input_receiver_fn


import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
from time import time
import numpy as np

tf.app.flags.DEFINE_string('server', 'ha05:8555',
                           'Server host:port.')
tf.app.flags.DEFINE_string('model', 'iris',
                           'Model name.')
FLAGS = tf.app.flags.FLAGS


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(_):
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model
    request.model_spec.signature_name = 'predict'

    p = []
    for i in range(1000):
        p.append(np.random.random())

    request.inputs['SepalLength'].CopyFrom(
        tf.make_tensor_proto(p, shape=[len(p)]))
    request.inputs['SepalWidth'].CopyFrom(
        tf.make_tensor_proto(p, shape=[len(p)]))
    request.inputs['PetalLength'].CopyFrom(
        tf.make_tensor_proto(p, shape=[len(p)]))
    request.inputs['PetalWidth'].CopyFrom(
        tf.make_tensor_proto(p, shape=[len(p)]))
    start = time()
    result_future = stub.Predict.future(request, 5.0)
    elapsed = (time() - start)
    prediction = result_future.result().outputs['probabilities']
    print(prediction)
    print("Time used:{0}ms".format(round(elapsed * 1000, 2)))


if __name__ == '__main__':
    tf.app.run()
