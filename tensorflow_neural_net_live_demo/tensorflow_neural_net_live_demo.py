"""
A Convolutional Network implementation example using TensorFlow library.

This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/).
"""
import tensorflow

class NeuralNet:
    """
    DOCSTRING
    """
    def __init__(self):
        self.batch_size = 128
        self.display_step = 10
        self.dropout = 0.75
        self.learning_rate = 0.001
        self.mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets('data', one_hot=True)
        self.n_classes = 10
        self.n_input = 784
        self.training_iters = 200000
        self.x = tensorflow.placeholder(tensorflow.float32, [None, self.n_input])
        self.y = tensorflow.placeholder(tensorflow.float32, [None, self.n_classes])
        self.keep_prob = tensorflow.placeholder(tensorflow.float32)

    def __call__(self):
        weights = {
            'wc1': tensorflow.Variable(tensorflow.random_normal([5, 5, 1, 32])),
            'wc2': tensorflow.Variable(tensorflow.random_normal([5, 5, 32, 64])),
            'wd1': tensorflow.Variable(tensorflow.random_normal([7*7*64, 1024])),
            'out': tensorflow.Variable(tensorflow.random_normal([1024, self.n_classes]))}
        biases = {
            'bc1': tensorflow.Variable(tensorflow.random_normal([32])),
            'bc2': tensorflow.Variable(tensorflow.random_normal([64])),
            'bd1': tensorflow.Variable(tensorflow.random_normal([1024])),
            'out': tensorflow.Variable(tensorflow.random_normal([self.n_classes]))}
        pred = self.conv_net(self.x, weights, biases, self.keep_prob)
        cost = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(pred, self.y))
        optimizer = tensorflow.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        correct_pred = tensorflow.equal(tensorflow.argmax(pred, 1), tensorflow.argmax(self.y, 1))
        accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_pred, tensorflow.float32))
        init = tensorflow.initialize_all_variables()
        with tensorflow.Session() as sess:
            sess.run(init)
            step = 1
            while step * self.batch_size < self.training_iters:
                batch_x, batch_y = self.mnist.train.next_batch(self.batch_size)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: self.dropout})
                if step % self.display_step == 0:
                    loss, acc = sess.run(
                        [cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    print('Iter {}, Minibatch Loss={:.6f}, Training Accuracy={:.5f}'.format(
                        str(step * self.batch_size), loss, acc))
                step += 1
            print('Optimization Finished!')
            print('Testing Accuracy:', sess.run(accuracy, feed_dict={
                x: self.mnist.test.images[:256], y: self.mnist.test.labels[:256], keep_prob: 1.0}))

    def conv_net(self, x, weights, biases, dropout):
        """
        DOCSTRING
        """
        x = tensorflow.reshape(x, shape=[-1, 28, 28, 1])
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        conv1 = self.maxpool2d(conv1, k=2)
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = self.maxpool2d(conv2, k=2)
        fc1 = tensorflow.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tensorflow.add(tensorflow.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tensorflow.nn.relu(fc1)
        fc1 = tensorflow.nn.dropout(fc1, dropout)
        out = tensorflow.add(tensorflow.matmul(fc1, weights['out']), biases['out'])
        return out

    def conv2d(self, x, W, b, strides=1):
        """
        DOCSTRING
        """
        x = tensorflow.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tensorflow.nn.bias_add(x, b)
        return tensorflow.nn.relu(x)

    def maxpool2d(self, x, k=2):
        """
        DOCSTRING
        """
        return tensorflow.nn.max_pool(
            x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
