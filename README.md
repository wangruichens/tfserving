# TF Serving 介绍、部署和Demo
---

tf serving:
支持模型热更新
支持版本管理
扩展性较好
稳定性，性能较好

### 一般工作流：

1、hdfs上的数据，使用spark/mapreduce/hive 进行数据分析和预处理

2、sub sample一部分数据，选择一个模型，预训练初始参数，交叉验证

3、使用全部数据集，spark to tfrecord 使用单机读取hdfs数据训练 or 多机多卡分布式训练

4、serving the model

# 一些解决方案：
---

## 方案1： yarn 3.1+ ： 
可以支持docker_image, [还不能提供稳定性保障](https://hadoop.apache.org/docs/r3.1.1/hadoop-yarn/hadoop-yarn-site/DockerContainers.html)

![image](https://github.com/wangruichens/samples/blob/master/distribute/tf/serving/serving.png)

[Docker+GPU support + tf serving + hadoop 3.1](https://community.hortonworks.com/articles/231660/tensorflow-serving-function-as-a-service-faas-with.html)


## 方案2： 模型Serving & 同步 from 美团blog
[参考链接](https://gitbook.cn/books/5b3adc411166b9562e9af3f6/index.html)

### 训练：tfrecord存放在hdfs上 （训练时拉取到本地）
### 预测：线上预估方案

- 模型同步

我们开发了一个高可用的同步组件：用户只需要提供线下训练好的模型的 HDFS 路径，该组件会自动同步到线上服务机器上。该组件基于 HTTPFS 实现，它是美团离线计算组提供的 HDFS 的 HTTP 方式访问接口。同步过程如下：

    同步前，检查模型 md5 文件，只有该文件更新了，才需要同步。
    同步时，随机链接 HTTPFS 机器并限制下载速度。
    同步后，校验模型文件 md5 值并备份旧模型。
    
同步过程中，如果发生错误或者超时，都会触发报警并重试。依赖这一组件，我们实现了在 2min 内可靠的将模型文件同步到线上。

- 模型计算

主要的问题在于解决网络IO和计算性能。

    并发请求。一个请求会召回很多符合条件的广告。在客户端多个广告并发请求 TF Serving，可以有效降低整体预估时延。
    特征 ID 化。通过将字符串类型的特征名哈希到 64 位整型空间，可以有效减少传输的数据量，降低使用的带宽。
    定制的模型计算，针对性优化



## 方案3： Centos 7 + docker + tfserving (当前使用方案)

### 训练： 实现细节在[这里](https://github.com/wangruichens/samples/tree/master/distribute/tf/spark_tfrecord)

### 预测：线上预估方案

#### 1、prerequisit： 安装docker

使用tfserving的docker版本。不想去踩编译和GPU功能拓展的坑。 
```
# 1: 安装相关软件
sudo yum install -y yum-utils device-mapper-persistent-data lvm2
# 2: 添加软件源信息 (阿里镜像)
sudo yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
# 3: 更新并安装 Docker-CE
sudo yum makecache fast
sudo yum -y install docker-ce
# 4: 开启Docker服务
sudo service docker start
# 5: 关闭Docker服务
sudo service docker stop
```

#### 2、使用训练好的model, 使用hdfs上tfrecord数据训练的手写数字识别model. 具体参见我之前的[配置](https://github.com/wangruichens/samples/tree/master/distribute/tf/spark_tfrecord)

模型很简单，参数量大概138w. 通过hdfs上的tfrecord来训练，模型文件保存在hdfs上

![image](https://github.com/wangruichens/samples/blob/master/distribute/tf/serving/model_des.png)

#### 3、docker启动tf serving, 拉取hdfs model 到本地并加载模型

[参考链接](https://www.tensorflow.org/tfx/serving/docker#serving_with_docker)

模型保存后的文件路径：

```
test_serving 
└───mnist_model_for_serving
│   └───1
│       │   variables
│       │   saved_model.pb
```
可以到模型目录/1/下查看模型输入输出：
```
saved_model_cli show --dir . --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_image'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 28, 28, 1)
        name: Conv1_input:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['Softmax/Softmax:0'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 10)
        name: Softmax/Softmax:0
  Method name is: tensorflow/serving/predict

```
docker 启动服务：

```
# For gRpc，默认端口8500
docker run -p 8500:8500 --mount type=bind,source=/home/wangrc/test_serving/mnist_model_for_serving,target=/models/mnist -e MODEL_NAME=mnist -t tensorflow/serving

# For REST, 默认端口8501
docker run -p 8501:8501 --mount type=bind,source=/home/wangrc/test_serving/mnist_model_for_serving,target=/models/mnist -e MODEL_NAME=mnist -t tensorflow/serving
```
当然也可以都启用这两个端口，-p 8500:8500 -p 8501:8501，也可以添加一些自己的config, 细节参考官方文档。

实际在模型中执行的命令是：
```
tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=my_model --model_base_path=/models/my_model
```

服务启动，模型成功加载：

![image](https://github.com/wangruichens/samples/blob/master/distribute/tf/serving/serving2.png)

查看端口占用和服务状态：
```
sudo netstat -nap | grep 8501
curl http://localhost:8501/v1/models/mnist
```

#### 4、使用REST测试模型结果（线上采用gRpc）：
```
python ./demo2/make_request.py
```
![image](https://github.com/wangruichens/samples/blob/master/distribute/tf/serving/res.png)


## 方案4： Centos 7 + tf serving + GPU without Docker

愿意踩坑的可以自己使用bazel编译：[参考链接](https://www.dearcodes.com/index.php/archives/25/)

# tf serving 使用nginx部署负载均衡 
[这里](https://github.com/wangruichens/samples/tree/master/distribute/tf/serving/serving_nginx)

# 主要问题：

tensorflow serving nginx 单个服务还能管理。当服务多起来，或者有服务间交互调用等问题时，再使用nginx config来管理就不现实了。需要一套类似spring cloud 的微服务框架。 而tf serving 只有python api， 并没有办法注册到spring cloud 来管理。

虽然tensorflow serving用于生产环境部署训练好的模型，但需要自己实现集群功能和健康检查，同时和java应用中间还隔着一个网络通讯的开销。所以最好还是java应用内部直接调用模型。python构建并训练模型+java在线预测还是比较合理的方案。

对于tensorflow来说，模型上线一般选择tensorflow serving或者client API库来上线，前者适合于较大的模型和应用场景，后者则适合中小型的模型和应用场景。因此算法工程师使用在产品之前需要做好选择和评估。

最合适的方案还是:k8s+docker。容器即服务的概念