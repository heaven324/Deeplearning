import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import loader3

#데이터 로드
train_image = "C:\\Users\\heaven\\python_project\\test_data\\total_image_32"
train_label = "C:\\Users\\heaven\\python_project\\test_data\\label1.csv"
test_image  = "C:\\Users\\heaven\\python_project\\test_data\\test_image_32"
test_label  = "C:\\Users\\heaven\\python_project\\test_data\\label1.csv"



print("LOADING DATA")

trainX = loader3.image_load(train_image)
trainY = loader3.label_load(train_label)
testX = loader3.image_load(test_image)
testY = loader3.label_load(test_label)

print("LOADED DATA")

tf.reset_default_graph()

#입력층
x = tf.placeholder("float",[None,32,32,3])
keep_prob = tf.placeholder("float")
training = tf.placeholder(tf.bool, name='training' )


#conv_1
b1 = tf.Variable(tf.ones([128]))
W1 = tf.Variable(tf.random_normal([3,3,3,128],stddev = 0.01))
y1 = tf.nn.conv2d(x, W1, strides=[1,1,1,1], padding = 'SAME')
y1 = y1 + b1
y1 = tf.contrib.layers.batch_norm(y1,scale=True, is_training=training)
y1 = tf.nn.leaky_relu(y1)

#conv_2
b1_2 = tf.Variable(tf.ones([128]))
W1_2 =  tf.Variable(tf.random_normal([3,3,128,128],stddev = 0.01))
y1_2 = tf.nn.conv2d(y1, W1_2, strides=[1,1,1,1], padding = 'SAME')
y1_2 = y1_2 + b1_2
y1_2 = tf.contrib.layers.batch_norm(y1_2,scale=True, is_training=training)
y1_2 = tf.nn.leaky_relu(y1_2)
y1_2 = tf.nn.dropout(y1_2, keep_prob)

#maxpooling
y1_2 = tf.nn.max_pool(y1_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
# 16 128

#conv_3
b2 = tf.Variable(tf.ones([256]))
W2 = tf.Variable(tf.random_normal([3,3,128,256],stddev = 0.01))
y2 = tf.nn.conv2d(y1_2, W2, strides=[1,1,1,1], padding = 'SAME')
y2 = y2 + b2
y2 = tf.contrib.layers.batch_norm(y2,scale=True, is_training=training)
y2 = tf.nn.leaky_relu(y2)

#conv_4
b2_2 = tf.Variable(tf.ones([256]))
W2_2 = tf.Variable(tf.random_normal([3,3,256,256],stddev = 0.01))
y2_2 = tf.nn.conv2d(y2, W2_2, strides=[1,1,1,1], padding = 'SAME')
y2_2 = y2_2 + b2_2
y2_2 = tf.contrib.layers.batch_norm(y2_2,scale=True, is_training=training)
y2_2 = tf.nn.leaky_relu(y2_2)
y2_2 = tf.nn.dropout(y2_2, keep_prob)

#maxpooling
y2_2 = tf.nn.max_pool(y2_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
# 8, 256

#conv_5
b3 = tf.Variable(tf.ones([512]))
W3 = tf.Variable(tf.random_normal([3,3,256,512],stddev = 0.01))
y3 = tf.nn.conv2d(y2_2, W3, strides=[1,1,1,1], padding = 'SAME')
y3 = y3 + b3
y3 = tf.contrib.layers.batch_norm(y3,scale=True, is_training=training)
y3 = tf.nn.leaky_relu(y3)

#conv_6
b3_2 = tf.Variable(tf.ones([512]))
W3_2 = tf.Variable(tf.random_normal([3,3,512,512],stddev = 0.01))
y3_2 = tf.nn.conv2d(y3, W3_2, strides=[1,1,1,1], padding = 'SAME')
y3_2 = y3_2 + b3_2
y3_2 = tf.contrib.layers.batch_norm(y3_2,scale=True, is_training=training)
y3_2 = tf.nn.leaky_relu(y3_2)
y3_2 = tf.nn.dropout(y3_2, keep_prob)

#maxpooling
y3_2 = tf.nn.max_pool(y3_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
# 4, 512

#Affine1
b4 = tf.Variable(tf.ones([1024]))
W4 = tf.get_variable(name='W4', shape=[4*4*512, 1024], initializer=tf.contrib.layers.variance_scaling_initializer())
y4 = tf.reshape(y3_2, [-1, 4*4*512])
y4 = tf.matmul(y4,W4) + b4
y4 = tf.contrib.layers.batch_norm(y4,scale=True, is_training=training)
y4 = tf.nn.leaky_relu(y4)


#Affine2
b5 = tf.Variable(tf.ones([1024]))
W5 = tf.get_variable(name='W5', shape=[1024, 1024], initializer=tf.contrib.layers.variance_scaling_initializer())
y5 = tf.matmul(y4,W5) + b5
y5 = tf.contrib.layers.batch_norm(y5,scale=True, is_training=training)
y5 = tf.nn.leaky_relu(y5)


#드롭아웃
y5_drop = tf.nn.dropout(y5, keep_prob)


#출력층
b6 = tf.Variable(tf.ones([25]))
W6 = tf.get_variable(name='W6', shape=[1024, 25], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 초기값
y6 = tf.matmul(y5_drop,W6) + b6
y6 = tf.contrib.layers.batch_norm(y6,scale=True, is_training=training)
y_hat = tf.nn.softmax(y6)


#예측값
y_predict = tf.argmax(y_hat,1)


# 라벨을 저장하기 위한 변수 생성
y_onehot = tf.placeholder("float",[None,25])
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
    for j in range(5000):
        trainX , trainY = loader3.shuffle_batch(trainX, trainY)
        testX, testY = loader3.shuffle_batch(testX, testY)
        for i in range(66):
            train_xs, train_ys = loader3.next_batch(trainX, trainY, i*100, i*100+100)
            test_xs, test_ys = loader3.next_batch(testX, testY, i*100, i*100+101)

            sess.run(Train, feed_dict={x: train_xs, y_onehot: train_ys, keep_prob: 0.8, training:True})

            if i == 0:
                train_acc = sess.run(accuracy, feed_dict={x: train_xs, y_onehot: train_ys, keep_prob: 1.0, training:False})
                test_acc = sess.run(accuracy, feed_dict={x: test_xs, y_onehot: test_ys, keep_prob: 1.0, training:False})

                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

                print('훈련  ', str(j + 1) + '에폭 정확도 :', train_acc)
                print('테스트', str(j + 1) + '에폭 정확도 :', test_acc)
                print('-----------------------------------------------')

# 모델 저장
    saver.save(sess, 'C:\\Users\\heaven\\python_project\\test_data\\model\\model')
                
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
