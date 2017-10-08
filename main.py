import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time


# Get the log directory
LOG_DIR = os.getcwd() + "/logs"

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    #Added by Anshoo

    #Load Model from path it is stored at ..
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    #Load the graph
    graph = tf.get_default_graph()
    
    #Extract Layers
    img_input_graph_node = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_graph_node = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    ly3_graph_node = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    ly4_graph_node = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    ly7_graph_node = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
       
    return img_input_graph_node, keep_prob_graph_node, ly3_graph_node, ly4_graph_node, ly7_graph_node
    #End Add by Anshoo
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # Add by Anshoo

# ENCODER PORTION:
    # On layer 7 which is the last layer prior Fully Connected in VGG
    # apply 1x1 conv to preserve spatial information,
    # this would complete our ENCODER
    ly7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 
                                    padding='SAME',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

# DECODER PORTION
    # On 1x1 conv, apply upsample / transpose 
    ly7_upsample_output = tf.layers.conv2d_transpose(ly7_conv_1x1, num_classes, 4, (2,2),
                                                     padding='SAME',
                                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # Skip Connection to layer 4
    # 1x1 of Layer 4 to ensure shapes are same when added next
    ly4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 
                                    padding='SAME',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # Bit wise addition of last layer output & layer 4
    skp4_add = tf.add(ly7_upsample_output, ly4_conv_1x1)
    skp4_upsample_output = tf.layers.conv2d_transpose(skp4_add, num_classes, 4, (2,2),
                                                      padding='same',
                                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                      kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
        
    # Skip Connection to layer 3
    # 1x1 of Layer 3 to ensure shapes are same when added next
    ly3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 
                                    padding='SAME',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # Bit wise addition of last layer output & layer 3
    skp3_add = tf.add(skp4_upsample_output, ly3_conv_1x1)
    skp3_upsample_output = tf.layers.conv2d_transpose(skp3_add, num_classes, 16, (8,8),
                                                      padding='SAME',
                                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                      kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    nn_last_layer = skp3_upsample_output
    return nn_last_layer
    #End Add by Anshoo
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

# Add by Anshoo
    # Reshape logits to be 2D Tensor where each row represent pixels & each column a class 
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # Reshape labels to match logits
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # Define loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))

    # Define optimizer be ADAM Optimizer
    optmizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

    # Define training operation
    trng_operation = optmizer.minimize(cross_entropy_loss)

    return logits, trng_operation, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

# Add by Anshoo
    #For tensorboard
    summary_op = tf.summary.merge_all()
    sum_writer = tf.summary.FileWriter(LOG_DIR + "/model" + "_" + str(time.time()))
    sum_writer.add_graph(sess.graph)

    # Initialize Global Variables
    sess.run(tf.global_variables_initializer())

    print()
    print ("Training commenced ..")
    print()

    start_time = time.clock()

    count = 1

    # Loop through Epocs
    for epoch in range(epochs):
        print()
        print (" Executing Epoch {}/{} ..".format(epoch+1, epochs))
        
        # Loop through batches, batches are carved for memory optimization
        for image, label in get_batches_fn(batch_size):
            if count % 20 == 0:
                _, loss, tfsumm = sess.run( [train_op, cross_entropy_loss, summary_op],
                                    feed_dict={input_image: image, 
                                           correct_label: label,
                                           keep_prob: 0.5,
                                           learning_rate: 1e-4} )
                # Write tensorboard summary 
                sum_writer.add_summary(tfsumm, count)
            else:
                _, loss = sess.run( [train_op, cross_entropy_loss],
                                    feed_dict={input_image: image, 
                                           correct_label: label,
                                           keep_prob: 0.5,
                                           learning_rate: 1e-4} )


            print ("  Loss per batch {:.3f}".format(loss))
            count += 1
    
    print()
    end_time = time.clock()
    train_time = end_time-start_time
    print ("Total time for training: {} secs".format(train_time))
    print()
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Path for Log folder to store tensorboard logs
    if os.path.exists(LOG_DIR + "/model*"):
        os.remove(LOG_DIR + "/model*")

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        epochs = 50
        batch_size = 5

        # Placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # Load VGG and retrieve layers from VGG
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        
        # Improvise VGG to FCN, applying Encoder & Decoder layers 
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # Loss Function & Optimizer 
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)


        # TODO: Train NN using the train_nn function

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, 
                 cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
