import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import loader3

#데이터 로드
train_image = 'C:\\Users\\heaven\\Desktop\\test\\vgg16_test_resize'
train_label = 'C:\\Users\\heaven\\Desktop\\test\\label.csv'
test_image = 'C:\\Users\\heaven\\Desktop\\test\\vgg16_test_resize'
test_label = 'C:\\Users\\heaven\\Desktop\\test\\label.csv'



print("LOADING DATA")

trainX = loader3.image_load(train_image)
trainY = loader3.label_load(train_label)
testX = loader3.image_load(test_image)
testY = loader3.label_load(test_label)

print("LOADED DATA")

tf.reset_default_graph()

#입력층
x = tf.placeholder("float",[None,224,224,3])
keep_prob = tf.placeholder("float")
training = tf.placeholder(tf.bool, name='training' )


#conv_1-1
b1 = tf.Variable(tf.ones([64]))
W1 = tf.Variable(tf.random_normal([3,3,3,64],stddev = 0.01))
y1 = tf.nn.conv2d(x, W1, strides=[1,1,1,1], padding = 'SAME')
y1 = y1 + b1
y1 = tf.contrib.layers.batch_norm(y1,scale=True, is_training=training)
y1 = tf.nn.leaky_relu(y1)

#conv_1-2
b1_2 = tf.Variable(tf.ones([64]))
W1_2 =  tf.Variable(tf.random_normal([3,3,64,64],stddev = 0.01))
y1_2 = tf.nn.conv2d(y1, W1_2, strides=[1,1,1,1], padding = 'SAME')
y1_2 = y1_2 + b1_2
y1_2 = tf.contrib.layers.batch_norm(y1_2,scale=True, is_training=training)
y1_2 = tf.nn.leaky_relu(y1_2)
y1_2 = tf.nn.dropout(y1_2, keep_prob)

#maxpooling
y1_2 = tf.nn.max_pool(y1_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
#[None,114,114,64]


#conv_2-1
b2 = tf.Variable(tf.ones([128]))
W2 = tf.Variable(tf.random_normal([3,3,64,128],stddev = 0.01))
y2 = tf.nn.conv2d(y1_2, W2, strides=[1,1,1,1], padding = 'SAME')
y2 = y2 + b2
y2 = tf.contrib.layers.batch_norm(y2,scale=True, is_training=training)
y2 = tf.nn.leaky_relu(y2)

#conv_2-2
b2_2 = tf.Variable(tf.ones([128]))
W2_2 = tf.Variable(tf.random_normal([3,3,128,128],stddev = 0.01))
y2_2 = tf.nn.conv2d(y2, W2_2, strides=[1,1,1,1], padding = 'SAME')
y2_2 = y2_2 + b2_2
y2_2 = tf.contrib.layers.batch_norm(y2_2,scale=True, is_training=training)
y2_2 = tf.nn.leaky_relu(y2_2)
y2_2 = tf.nn.dropout(y2_2, keep_prob)

#maxpooling
y2_2 = tf.nn.max_pool(y2_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
#[None,56,56,128]


#conv_3-1
b3 = tf.Variable(tf.ones([256]))
W3 = tf.Variable(tf.random_normal([3,3,128,256],stddev = 0.01))
y3 = tf.nn.conv2d(y2_2, W3, strides=[1,1,1,1], padding = 'SAME')
y3 = y3 + b3
y3 = tf.contrib.layers.batch_norm(y3,scale=True, is_training=training)
y3 = tf.nn.leaky_relu(y3)

#conv_3-2
b3_2 = tf.Variable(tf.ones([256]))
W3_2 = tf.Variable(tf.random_normal([3,3,256,256],stddev = 0.01))
y3_2 = tf.nn.conv2d(y3, W3_2, strides=[1,1,1,1], padding = 'SAME')
y3_2 = y3_2 + b3_2
y3_2 = tf.contrib.layers.batch_norm(y3_2,scale=True, is_training=training)
y3_2 = tf.nn.leaky_relu(y3_2)
y3_2 = tf.nn.dropout(y3_2, keep_prob)

#conv_3-3
b3_3 = tf.Variable(tf.ones([256]))
W3_3 = tf.Variable(tf.random_normal([3,3,256,256],stddev = 0.01))
y3_3 = tf.nn.conv2d(y3_2, W3_3, strides=[1,1,1,1], padding = 'SAME')
y3_3 = y3_3 + b3_3
y3_3 = tf.contrib.layers.batch_norm(y3_3,scale=True, is_training=training)
y3_3 = tf.nn.leaky_relu(y3_3)
y3_3 = tf.nn.dropout(y3_3, keep_prob)

#maxpooling
y3_3 = tf.nn.max_pool(y3_3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
#[None,28,28, 256]


#conv_4-1
b4 = tf.Variable(tf.ones([512]))
W4 = tf.Variable(tf.random_normal([3,3,256,512],stddev = 0.01))
y4 = tf.nn.conv2d(y3_3, W4, strides=[1,1,1,1], padding = 'SAME')
y4 = y4 + b4
y4 = tf.contrib.layers.batch_norm(y4,scale=True, is_training=training)
y4 = tf.nn.leaky_relu(y4)

#conv_4-2
b4_2 = tf.Variable(tf.ones([512]))
W4_2 = tf.Variable(tf.random_normal([3,3,512,512],stddev = 0.01))
y4_2 = tf.nn.conv2d(y4, W4_2, strides=[1,1,1,1], padding = 'SAME')
y4_2 = y4_2 + b4_2
y4_2 = tf.contrib.layers.batch_norm(y4_2,scale=True, is_training=training)
y4_2 = tf.nn.leaky_relu(y4_2)
y4_2 = tf.nn.dropout(y4_2, keep_prob)

#conv_4-3
b4_3 = tf.Variable(tf.ones([512]))
W4_3 = tf.Variable(tf.random_normal([3,3,512,512],stddev = 0.01))
y4_3 = tf.nn.conv2d(y4_2, W4_3, strides=[1,1,1,1], padding = 'SAME')
y4_3 = y4_3 + b4_3
y4_3 = tf.contrib.layers.batch_norm(y4_3,scale=True, is_training=training)
y4_3 = tf.nn.leaky_relu(y4_3)
y4_3 = tf.nn.dropout(y4_3, keep_prob)

#maxpooling
y4_3 = tf.nn.max_pool(y4_3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
#[None,14,14,512]


#conv_5-1
b5 = tf.Variable(tf.ones([512]))
W5 = tf.Variable(tf.random_normal([3,3,512,512],stddev = 0.01))
y5 = tf.nn.conv2d(y4_3, W5, strides=[1,1,1,1], padding = 'SAME')
y5 = y5 + b5
y5 = tf.contrib.layers.batch_norm(y5,scale=True, is_training=training)
y5 = tf.nn.leaky_relu(y5)

#conv_5-2
b5_2 = tf.Variable(tf.ones([512]))
W5_2 = tf.Variable(tf.random_normal([3,3,512,512],stddev = 0.01))
y5_2 = tf.nn.conv2d(y5, W5_2, strides=[1,1,1,1], padding = 'SAME')
y5_2 = y5_2 + b5_2
y5_2 = tf.contrib.layers.batch_norm(y5_2,scale=True, is_training=training)
y5_2 = tf.nn.leaky_relu(y5_2)
y5_2 = tf.nn.dropout(y5_2, keep_prob)

#conv_5-3
b5_3 = tf.Variable(tf.ones([512]))
W5_3 = tf.Variable(tf.random_normal([3,3,512,512],stddev = 0.01))
y5_3 = tf.nn.conv2d(y5_2, W5_3, strides=[1,1,1,1], padding = 'SAME')
y5_3 = y5_3 + b5_3
y5_3 = tf.contrib.layers.batch_norm(y5_3,scale=True, is_training=training)
y5_3 = tf.nn.leaky_relu(y5_3)
y5_3 = tf.nn.dropout(y5_3, keep_prob)

#maxpooling
y5_3 = tf.nn.max_pool(y5_3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
#[None,7,7,512]


#Affine1
b6 = tf.Variable(tf.ones([4096]))
W6 = tf.get_variable(name='W6', shape=[7*7*512, 4096], initializer=tf.contrib.layers.variance_scaling_initializer())
y6 = tf.reshape(y5_3, [-1, 7*7*512])
y6 = tf.matmul(y6,W6) + b6
y6 = tf.contrib.layers.batch_norm(y6,scale=True, is_training=training)
y6 = tf.nn.leaky_relu(y6)


#Affine2
b7 = tf.Variable(tf.ones([1000]))
W7 = tf.get_variable(name='W7', shape=[4096, 1000], initializer=tf.contrib.layers.variance_scaling_initializer())
y7 = tf.matmul(y6,W7) + b7
y7 = tf.contrib.layers.batch_norm(y7,scale=True, is_training=training)
y7 = tf.nn.leaky_relu(y7)


#드롭아웃
y7_drop = tf.nn.dropout(y7, keep_prob)


#출력층
b8 = tf.Variable(tf.ones([2]))
W8 = tf.get_variable(name='W8', shape=[1000, 2], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 초기값
y8 = tf.matmul(y7_drop,W8) + b8
y8 = tf.contrib.layers.batch_norm(y8,scale=True, is_training=training)
y_hat = tf.nn.softmax(y8)


#예측값
y_predict = tf.argmax(y_hat,1)


# 라벨을 저장하기 위한 변수 생성
y_onehot = tf.placeholder("float",[None,2])
y_label = tf.argmax(y_onehot, axis = 1)


# 정확도를 출력하기 위한 변수 생성
correct_prediction = tf.equal(y_predict, y_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))


# 교차 엔트로피 오차 함수
loss = -tf.reduce_sum(y_onehot * tf.log(y_hat), axis = 1)


# SGD 경사 감소법
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step
    Train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Adam 경사 감소법
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001)


# 학습 오퍼레이션 정의
# train = optimizer.minimize(loss)

# 모델 저장
saver = tf.train.Saver()

# 변수 초기화
init = tf.global_variables_initializer()
train_acc_list = []
test_acc_list = []


with tf.Session() as sess:
    sess.run(init)
    for j in range(20):
        for i in range(600):
            trainX , trainY = loader3.shuffle_batch(trainX, trainY)
            testX, testY = loader3.shuffle_batch(testX, testY)

            train_xs, train_ys = loader3.next_batch(trainX, trainY, 0, 2)
            test_xs, test_ys = loader3.next_batch(testX, testY, 0, 2)

            sess.run(Train, feed_dict={x: train_xs, y_onehot: train_ys, keep_prob: 0.8, training:True})

            if i == 0:
                train_acc = sess.run(accuracy, feed_dict={x: train_xs, y_onehot: train_ys, keep_prob: 1.0, training:False})
                test_acc = sess.run(accuracy, feed_dict={x: test_xs, y_onehot: test_ys, keep_prob: 1.0, training:False})

                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

                print('훈련', str(j + 1) + '에폭 정확도 :', train_acc)
                print('테스트', str(j + 1) + '에폭 정확도 :', test_acc)
                print('-----------------------------------------------')

# 모델 저장
    saver.save(sess, 'C:\\Users\\heaven\\Desktop\\test\\model\\model')
                
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot()
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(min(min(train_acc_list),min(test_acc_list))-0.1, 1.1)
plt.legend(loc='lower right')
plt.show()