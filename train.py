import numpy as np
import tensorflow as tf
import pic_process

pic_width = 100
pic_height = 100
batch_size = 100
learning_rate = 0.001


data_sets = pic_process.prepare_data()
images_placeholder = tf.placeholder(tf.float32, shape=[None, pic_width * pic_height * 3])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])
weights = tf.Variable(tf.zeros([pic_width * pic_height * 3, 2]))
biases = tf.Variable(tf.zeros([2]))

logits = tf.matmul(images_placeholder, weights) + biases
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(2000):
        indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
        images_batch = data_sets['images_train'][indices]
        labels_batch = data_sets['labels_train'][indices]

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                images_placeholder: images_batch, labels_placeholder: labels_batch})
            print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))
        sess.run(train_step, feed_dict={images_placeholder: images_batch, labels_placeholder: labels_batch})

    test_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: data_sets['images_test'],
        labels_placeholder: data_sets['labels_test']})
    print('Test accuracy {:g}'.format(test_accuracy))
