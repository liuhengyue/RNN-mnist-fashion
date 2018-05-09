## import required library here

import random
import numpy as np
import tensorflow as tf
import pdb
import tensorflow.examples.tutorials.mnist.input_data as input_data

seed_value = 42
tf.set_random_seed(seed_value)
random.seed(seed_value)

## step 1: prepare data set

mnist = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=True)
# define true rate of model
#train_x = mnist.train.next_batch(40)

## step 2: building rnn training model

# Hyper-parameters
hidden_size   = 100  # hidden layer's size
learning_rate = 1e-1
picture_size = 784
step = 28
pixel_size = 28
class_size = 10

# input layer
inputs     = tf.placeholder(shape=[None, pixel_size], dtype=tf.float32, name="inputs")
targets    = tf.placeholder(shape=[None, class_size], dtype=tf.float32, name="targets")
init_state = tf.placeholder(shape=[None, hidden_size], dtype=tf.float32, name="state")

initializer = tf.random_normal_initializer(stddev=0.1)

#tensorboard
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

# define rnn model
sess = tf.Session()
with tf.variable_scope("RNN") as scope:
    hs_t = init_state
    ys = []
    for i, pixel in enumerate(tf.split(inputs, step, axis=0)):
        if i > 0: scope.reuse_variables()  # Reuse variables
        Wxh = tf.get_variable("Wxh", [pixel_size, hidden_size], initializer=initializer) # U
        Whh = tf.get_variable("Whh", [hidden_size, hidden_size], initializer=initializer) # W
        Why = tf.get_variable("Why", [hidden_size, class_size], initializer=initializer) # V
        bh  = tf.get_variable("bh", [hidden_size], initializer=initializer)
        by  = tf.get_variable("by", [class_size], initializer=initializer)

        hs_t = tf.tanh(tf.matmul(pixel, Wxh) + tf.matmul(hs_t, Whh) + bh)
        # only output on last pixel -- many to one
        if i == step - 1:
            ys_t = tf.matmul(hs_t, Why) + by
        # ys.append(ys_t)
hprev = hs_t
output_softmax = tf.nn.softmax(ys_t)  # Get softmax for sampling

## step 3: define training method -- eg.loss function, optimizer...\n",

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=output_softmax))

minimizer = tf.train.AdamOptimizer()
grads_and_vars = minimizer.compute_gradients(loss)
grad_clipping = tf.constant(5.0, name="grad_clipping")
clipped_grads_and_vars = []
for grad, var in grads_and_vars:
    clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
    clipped_grads_and_vars.append((clipped_grad, var))

#data.train.next_batch(BATCH_SIZE)

correct_prediction = tf.equal(tf.argmax(targets,1), tf.argmax(output_softmax, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

loss_function = tf.reduce_mean(
                   tf.nn.softmax_cross_entropy_with_logits
                       (logits=output_softmax ,
                        labels=targets))

# start training
trainEpochs = 200
batchSize = 40
totalBatchs = int(mnist.train.num_examples/batchSize)
epoch_list=[];accuracy_list=[];loss_list=[];
from time import time
startTime=time()


## step 2: building training model

## step 3: define training method -- eg.loss function, optimizer...\n",

## step 4: output the result of training

for epoch in range(trainEpochs):
    hprev_val = np.zeros([1, hidden_size])
    for i in range(totalBatchs):
        #_x, _y = sess.run([batch_x, batch_y])
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        batch_x = batch_x.reshape([-1,28])
        hprev_val, loss_val, _ = sess.run([hprev, loss, updates],
                                      feed_dict={inputs: batch_x,
                                                 targets: batch_y,
                                                 init_state: hprev_val})
    #print(test_yy)
    loss, acc = sess.run([loss_function, accuracy], feed_dict={inputs: mnist.validation.images.reshape([-1,28]), targets: mnist.validation.labels})

    epoch_list.append(epoch)
    loss_list.append(loss);accuracy_list.append(acc)

    print("Train Epoch:", '%02d' % (epoch+1), \
          "Loss=","{:.9f}".format(loss)," Accuracy=",acc)

duration =time()-startTime
print("Train Finished takes:",duration)
