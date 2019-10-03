import tensorflow as tf
import cnn

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('\n Start Session! \n')
for i in range(1000):
    batch = cnn.mnist.train.next_batch(100)
    train_accuracy = sess.run(cnn.accuracy, \
                              feed_dict={cnn.x:batch[0], cnn.y_:batch[1], cnn.keep_prob:1.0})
    print('\r loop : {}   accuracy : {}   '.format(i, train_accuracy), end='')
    sess.run(cnn.train_step, feed_dict={cnn.x:batch[0], cnn.y_:batch[1], cnn.keep_prob:0.5})

print("\n test accuracy %g"% sess.run(cnn.accuracy, \
            feed_dict={cnn.x: cnn.mnist.test.images, \
                       cnn.y_: cnn.mnist.test.labels, cnn.keep_prob: 1.0}))
print('\n End Session!')