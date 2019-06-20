Nginx + TF Serving using Docker

为tf serving 配置负载均衡。

1、
首先使用docker pull最新的nginx 镜像, 将其中的nginx.conf拷贝出来，修改成 ./nginx.conf的格式，配置主从节点ip和端口。
* $upstream_addr ： log_format配置，当ngnix做负载均衡时，可以查看后台提供真实服务的设备

2、
在每台机器上启动tfserving服务：
```angular2
docker run -d -p 8501:8501 --mount type=bind,source=/home/wangrc/mnist_model_for_serving,target=/models/mnist -e MODEL_NAME=mnist -t tensorflow/serving
```

3、
在主节点 ha05 上执行：
```angular2
docker run -d -p 8256:80 -p 8255:8255 -v '/home/wangrc/mnist_log:/var/log/nginx' --name nginx_server mynginx
```
映射相关log到本地

4、 测试端口是否正常工作。 可以通过make_request.py提交post请求，查看log确认负载均衡搭建成功。
```angular2
curl http://ha05:8255/v1/models/mnist
```
