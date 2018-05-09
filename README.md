# Fashion-MNIST (classification with RNN)

[![Gitlab](https://img.shields.io/badge/Gitlab-fashion_mnist-blue.svg)](http://gitlab.icenter.tsinghua.edu.cn/qy-chen17/RNN-fashion-classification)
[![Source](https://img.shields.io/badge/Source-fashion_mnist-green.svg)](https://github.com/zalandoresearch/fashion-mnist/)
[![Data](https://img.shields.io/badge/Data-fashion_mnist-yellow.svg)](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/)
[![Tutorial_Source](https://img.shields.io/badge/Tutorial_Source-mnist-purple.svg)](https://www.youtube.com/watch?v=SeffmcG42SY&t=384s)


<details><summary>Table of Contents</summary><p>
* [Introduction](#introduction)
* [Get the Data](#get-the-data)
* [Using RNN to classify Fashion-MNIST](#using-rnn-to-classify-fashion-mnist)
</p></details><p></p>

## Introduction
`Fashion-MNIST` Fashion-MNIST是一个替代MNIST手写数字集的图像数据集。 它是由Zalando（一家德国的时尚科技公司）旗下的研究部门提供。其涵盖了来自10种类别的共7万个不同商品的正面图片。 我们的目的是利用课堂所学的RNN的方式将 `Fashion-MNIST` 分类，并观察其效果。


这个数据集的样子大致如下（每个类别占三行）：

![](https://kaggle2.blob.core.windows.net/datasets-images/2243/3791/9384af51de8baa77f6320901f53bd26b/data-original.png)


## Get the Data

我们从 [Zalando]('https://research.zalando.com/')的网站获取数据，代码如下：

```bash
mnist = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
```

### Labels
`Fashion-MNIST`将服饰分成10个种类并进行了标注，如下：

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |


## Using RNN to classify Fashion-MNIST


### Placeholder
placeholder code 如下：
```bash
inputs     = tf.placeholder(shape=[None, pixel_size], dtype=tf.float32, name="inputs")
targets    = tf.placeholder(shape=[None, class_size], dtype=tf.float32, name="targets")
init_state = tf.placeholder(shape=[None, hidden_size], dtype=tf.float32, name="state")
```
因为`Fashion-MNIST`的image shape是 28*28，表示rnn 会在一张图中跑28次(step=28)，而每次跑的一行都会有28个column(pixel_size=28).
另外，上述提及总共有10个种类(class_size=10)

```bash
# Hyper-parameters
hidden_size   = 100  # hidden layer's size
learning_rate = 1e-1
picture_size = 784
step = 28
pixel_size = 28
class_size = 10
```
