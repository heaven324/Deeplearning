import Basic_linear_regression as blr
import tensorflow as tf
import matplotlib.pyplot as plt

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * blr.x_data + b

# Gradient_descent 정의
loss = tf.reduce_mean(tf.square(y - blr.y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 변수, 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 알고리즘 실행(8반복)

print('------------Session Start------------')

for step in range(8):
    sess.run(train)
    
    # 변수 변화량 확인
    print(step, sess.run(W), sess.run(b))

    # 변화량 그래프로 확인
    plt.plot(step, sess.run(W), sess.run(b), sess.run(loss))
    plt.plot(blr.x_data, blr.y_data, 'ro')
    plt.plot(blr.x_data, sess.run(W) * blr.x_data + sess.run(b))
    plt.xlabel('x')
    plt.xlim(-2, 2)
    plt.ylim(0.1, 0.6)
    plt.ylabel('y')
    plt.show()
