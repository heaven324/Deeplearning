import kmean_create_data as kcd

import tensorflow as tf

# 데이터를 텐서로 옮기기
vectors = tf.constant(kcd.vectors_set)


# k개의 중심 선택
k = 4
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [k,-1]))


# 연산을 위한 차원 확장
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)


# 유클리드 제곱 거리 할당 1단계
'''
d^2(vector, centroid) = (vector_x - centroid_x)^2 + (vector_y - centroid_y)^2
'''
diff = tf.subtract(expanded_vectors, expanded_centroids) # 뺄셈
sqr = tf.square(diff)  # 제곱
distances = tf.reduce_sum(sqr, 2)  # 차원 감소(차원에 따라 원소들을 더함)
assignments = tf.argmin(distances, 0)  # 가장 작은 값의 인덱스 리턴(데이터의 중심)
# 위 네줄을 한번에 한 코드
# assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, \
#                                                            expanded_centroids)), 2), 0)


# 새로운 중심 계산
means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)),
                                [1,-1])), reduction_indices = [1]) for c in range(k)], 0)


# 루프 구성 (중심값 업데이트)
update_centroids = tf.assign(centroids, means)


# 변수 초기화
init_op = tf.global_variables_initializer()


# 그래프 실행
sess = tf.Session()
sess.run(init_op)

for step in range(100):
    _, centroids_values, assignment_values = sess.run([update_centroids, centroids, assignments])
