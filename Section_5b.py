import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Running params
num_of_channels = 1
num_of_classes = 10
image_size = 28

drop_rate = 0.4
learning_rate = 0.001
batch_size = 100
num_of_iterations = 5000

# Network layers params
filter_size_conv1 = 5
num_filters_conv1 = 32

pool_size_max_pool1 = 2
stride_size_maxpool1 = 2

filter_size_conv2 = 5
num_filters_conv2 = 64

pool_size_max_pool2 = 2
stride_size_maxpool2 = 2

layer_size_fc1 = 1024

layer_size_fc2 = 1024

# Reading the data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Creating the session
session = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, num_of_channels], name='x')

y_true = tf.placeholder(tf.float32, shape=[None, num_of_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Creating a placeholder for the drop rate
keep_prob = tf.placeholder_with_default(1 - drop_rate, shape=(), name="keep_prob")


#################################### Creating layers functions ####################################

# Init weights by the given shape
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


# Init biases by the given size
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


# Create a max pooling layer
def create_maxPool(input, pool_size, stride, padding="VALID"):
    layer = tf.nn.max_pool(value=input, ksize=[1, pool_size, pool_size, 1], strides=[1, stride, stride, 1],
                           padding=padding)
    return layer


# Create convolutional 2-dimensional layer
def create_conv2d(input, num_input_channels, conv_filter_size, num_filters):
    # Define weights and biases
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)

    # Create convolutional layer using nn.conv2d
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add biases and apply activation
    layer += biases
    layer = tf.nn.relu(layer)

    return layer


# Create a flatten layer for a dense layer input
def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    return layer


# Create fully connected layer (Dense)
def create_fullyConnected_layer(input, num_inputs, num_outputs, use_relu=True):
    # Define weights and biases
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Calculate the weights multiplication and add biases
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# Create dropout layer
def create_dropout_layer(input, keep_prob):
    return tf.nn.dropout(x=input, keep_prob=keep_prob)


# Create normalization layer
def create_normalization_layer(input):
    return tf.layers.batch_normalization(input)


#################################### Building the network ####################################

layer_conv1 = create_conv2d(input=x,
                            num_input_channels=num_of_channels,
                            conv_filter_size=filter_size_conv1,
                            num_filters=num_filters_conv1)

layer_norm1= create_normalization_layer(layer_conv1)

layer_max_pooling1 = create_maxPool(input=layer_norm1,
                                    pool_size=pool_size_max_pool1,
                                    stride=stride_size_maxpool1)

layer_conv2 = create_conv2d(input=layer_conv1,
                            num_input_channels=num_filters_conv1,
                            conv_filter_size=filter_size_conv2,
                            num_filters=num_filters_conv2)

layer_norm2 = create_normalization_layer(layer_conv2)

layer_max_pooling2 = create_maxPool(input=layer_norm2,
                                    pool_size=pool_size_max_pool2,
                                    stride=stride_size_maxpool2)

layer_flat = create_flatten_layer(layer_max_pooling2)

layer_fc1 = create_fullyConnected_layer(input=layer_flat,
                                        num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                        num_outputs=layer_size_fc1,
                                        use_relu=True)

drop_layer = create_dropout_layer(layer_fc1, keep_prob)

layer_fc2 = create_fullyConnected_layer(input=drop_layer,
                                        num_inputs=layer_size_fc1,
                                        num_outputs=layer_size_fc2,
                                        use_relu=True)

layer_logits = create_fullyConnected_layer(input=layer_fc2,
                                           num_inputs=layer_size_fc2,
                                           num_outputs=num_of_classes,
                                           use_relu=False)

y_pred = tf.nn.softmax(layer_logits, name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)

session.run(tf.global_variables_initializer())

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_logits,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())


def print_log(iteration, train_cost):
    msg = "Training Iteration {0} ---> Training Cost: {1:.3f}"
    print(msg.format(iteration, train_cost))

val_accuracy = 0

def train(num_iteration, print_every_n=5):
    global val_accuracy
    no_improve_streak = 0
    acc_3_last = 0
    for i in range(num_iteration):

        batch = mnist.train.next_batch(batch_size)
        x_batch = np.reshape(batch[0], [-1, image_size, image_size, num_of_channels])

        feed_dict_tr = {x: x_batch, y_true: batch[1]}
        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % print_every_n == 0:
            train_cost = session.run(cost, feed_dict=feed_dict_tr)
            print_log(iteration=i, train_cost=train_cost)

        val_x = np.reshape(mnist.validation.images, [-1, image_size, image_size, num_of_channels])
        val_accuracy = session.run(accuracy, feed_dict={x: val_x, y_true: mnist.validation.labels})

        if val_accuracy >= acc_3_last:
            acc_3_last = val_accuracy
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        if no_improve_streak == 3:
            print("Training stopped after "+str(i)+" iterations because no improvement on validation set.")
            break


print("Start Training...")
train(num_iteration=num_of_iterations)
print("Finish Training!")
test_x = np.reshape(mnist.test.images, [-1, image_size, image_size, num_of_channels])
test_accuracy = session.run(accuracy, feed_dict={x: test_x, y_true: mnist.test.labels})
print("Validation Accuracy ---> " + str(val_accuracy))
print("Test Accuracy ---> " + str(test_accuracy))


session.close()
