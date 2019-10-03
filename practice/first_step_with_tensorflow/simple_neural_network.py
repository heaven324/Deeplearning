import tensorflow as tf


# 데이터 로드
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


# 데이터의 차원 확인
# tf.convert_to_tensor(mnist.train.images).get_shape()
# tf.convert_to_tensor(mnist.train.labels).get_shape()


# 가중치, 편향 변수 생성
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# 이미지 정보 텐서에 공간 할당
x = tf.placeholder("float", [None, 784])  # None : 어떤 크기도 가능하다라는 뜻


# softmax 함수
y = tf.nn.softmax(tf.matmul(x, W) + b)


# 라벨 정보 텐서에 공간 할당
y_ = tf.placeholder("float", [None, 10])


# cross_entropy 함수
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


# 역전파 알고리즘
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 모델 평가
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 세션 생성 및 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 세션 루프 설정
print('\n Start Session! \n')
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})
    print('\r loop : {}   accuracy : {}   '.format(i, \
          sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels})), end='')
print('\n End Session!')

