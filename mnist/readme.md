train model and save ->running service -> make request

docker中运行：
参考 ： https://www.tensorflow.org/tfx/serving/docker#serving_with_docker

启动cpu版本service
docker run -p 8500:8500 --mount type=bind,source=/home/wangrc/test_serving/mnist_model_for_serving,target=/models/mnist -e MODEL_NAME=mnist -t tensorflow/serving

启动gpu版本service
docker run --runtime=nvidia -p 8501:8501 --mount type=bind,source=/home/wangrc/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_gpu,target=/models/half_plus_two -e MODEL_NAME=half_plus_two -t tensorflow/serving:latest-gpu


文件夹目录：
mnist_model_for_serving/1/ saved_model.pb  variables 

saved_model_cli show --dir . --all

curl http://localhost:8501/v1/models/mnist
