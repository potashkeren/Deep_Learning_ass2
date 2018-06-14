

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)



def model_func(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Conv1
    # Input shape -  [batch_size, 28, 28, 1]
    # Output shape - [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)

    # Pool1
    # Input Shape - [batch_size, 28, 28, 32]
    # Output Shape - [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Conv2
    # Input Shape - [batch_size, 14, 14, 32]
    # Output Shape - [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size = 5,
        padding="same",
        activation=tf.nn.relu)

    # Pool2
    # Input Shape - [batch_size, 14, 14, 64]
    # Output Shape - [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64]) # Flattened tensor

    # Dense1
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Dense2
    # Input Shape - [batch_size, 1024]
    # Output Shape - [batch_size, 1024]
    dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu)

    # Logits
    # Input Shape - [batch_size, 1024]
    # Output Shape - [batch_size, 10]
    logits = tf.layers.dense(inputs=dense2, units=10)

    predictions = {
        # get class predicted by the model
        "classes": tf.argmax(input=logits, axis=1),
        # get softmax tensor (probabilities)
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    # Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss (AKA cost) by softmaxing the logits
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Train
    if mode == tf.estimator.ModeKeys.TRAIN:

        # define optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # define operation
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Evaluate
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_func, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    # tensors_to_log = {"probabilities": "loss_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=5000)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
