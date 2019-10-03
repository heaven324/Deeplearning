from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

import tensorflow as tf
import cnn_function as cnef

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


x_image = tf.reshape(x, [-1, 28, 28, 1])


# 매개변수 생성
W_conv1 = cnef.weight_variable([5, 5, 1, 32])
b_conv1 = cnef.bias_variable([32])

W_conv2 = cnef.weight_variable([5, 5, 32, 64])
b_conv2 = cnef.bias_variable([64])

W_fc1 = cnef.weight_variable([7*7*64, 1024])
b_fc1 = cnef.bias_variable([1024])

W_fc2 = cnef.weight_variable([1024, 10])
b_fc2 = cnef.bias_variable([10])

# 드롭아웃 공간 생성
keep_prob = tf.placeholder("float")


# 계층 생성
h_conv1 = tf.nn.relu(cnef.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = cnef.max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(cnef.conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = cnef.max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv =tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 학습 알고리즘(역전파, Adam)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 모델 평가
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 세션 생성 및 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# =============================================================================
# # 세션 루프 설정
# print('\n Start Session! \n')
# for i in range(100):
#     batch = mnist.train.next_batch(100)
#     train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
#     print('\r loop : {}   accuracy : {}   '.format(i, train_accuracy), end='')
#     sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
# 
# print("\n test accuracy %g"% sess.run(accuracy, \
#             feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# print('\n End Session!')
# 
# =============================================================================

