#!/usr/bin/env bash
tensorflow_model_server \
--port=8500 \
--rest_api_port=8501 \
--model_name=iris \
--model_base_path="/home/wangrc/github/summaries/serving/estimator/export_raw"



docker run -p 8555:8555 --mount type=bind,source=/home/wangrc/export_parsing,target=/models/iris -e MODEL_NAME=iris -t tensorflow/serving

docker run -p 8500:8500 --mount type=bind,source=/home/wangrc/export/,target=/models/deepfm -e MODEL_NAME=deepfm -t tensorflow/serving

curl -o - http://ha05:8500/v1/models/deepfm


curl -o - http://algorithmsdeepfm.2345.cn:38609/v1/models/deepfm