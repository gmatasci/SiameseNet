import tensorflow as tf

class SiameseNet:

    # Create model
    def __init__(self, is_image, comparison_type):

        self.is_training = tf.placeholder(tf.bool, shape=())

        if is_image:
            self.x1 = tf.placeholder(tf.float32, [None, 28, 28, 1])
            self.x2 = tf.placeholder(tf.float32, [None, 28, 28, 1])
            with tf.variable_scope("siamese") as scope:
                self.o1 = self.twin_block_CNN(self.x1, self.is_training)
                scope.reuse_variables()
                self.o2 = self.twin_block_CNN(self.x2, self.is_training)
        else:
            self.x1 = tf.placeholder(tf.float32, [None, 784])
            self.x2 = tf.placeholder(tf.float32, [None, 784])
            with tf.variable_scope("siamese") as scope:
                self.o1 = self.twin_block(self.x1, self.is_training)
                scope.reuse_variables()
                self.o2 = self.twin_block(self.x2, self.is_training)

        # Compare by subtraction or concatenation the output of the 2 twin networks
        if comparison_type == 'subtract':
            self.pair_logit, self.pair_prob = self.subtract_block()
        elif comparison_type == 'concat':
            self.pair_logit, self.pair_prob = self.concat_block()

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        # self.eucd, self.losses, self.pair_prob, self.loss = self.loss_with_spring()
        self.loss = self.loss_binary_cross_entr()



    def twin_block(self, x, is_training):

        with tf.variable_scope('twin_block', reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='fc1')
            # net = tf.layers.batch_normalization(net, center=True, scale=True, training=is_training)  # training will then take a value False when used in test to normalize test data by the overall mean on the training set
            net = tf.layers.dense(net, 1024, activation=tf.nn.relu, name='fc2')
            # net = tf.layers.batch_normalization(net, center=True, scale=True, training=is_training)  # training will then take a value False when used in test to normalize test data by the overall mean on the training set
            output = tf.layers.dense(net, 512, activation=tf.nn.relu, name='fc3')
            # output = tf.layers.batch_normalization(net, center=True, scale=True, training=is_training)  # training will then take a value False when used in test to normalize test data by the overall mean on the training set

        return output

    def twin_block_CNN(self, x, is_training):

        with tf.variable_scope('twin_block_CNN', reuse=tf.AUTO_REUSE):
            ## Convolutional Layer #1
            # Computes 64 features using a 5x5 filter with ReLU activation.
            net = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
            # net = tf.layers.batch_normalization(net, center=True, scale=True, training=is_training)  # training will then take a value False when used in test to normalize test data by the overall mean on the training set

            ## Pooling Layer #1
            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

            ## Convolutional Layer #2
            net = tf.layers.conv2d(net, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
            # net = tf.layers.batch_normalization(net, center=True, scale=True, training=is_training)

            ## Pooling Layer #2
            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

            ## Fully connected (dense) Layer
            net = tf.reshape(net, [-1, 7*7*32])
            net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu)

            ## Dropout layer: will only be performed if training is True
            net = tf.layers.dropout(inputs=net, rate=0.3, training=is_training)

            ## Fully connected (dense) Layer
            output = tf.layers.dense(inputs=net, units=512, activation=None)

        return output

    def subtract_block(self):

        # Initialze neural network
        with tf.variable_scope("subtract_block") as scope:

            substr = tf.subtract(self.o1, self.o2)
            fc1 = tf.layers.dense(substr, 1024, activation=tf.nn.relu, name='fc1')
            logits = tf.layers.dense(fc1, 1, activation=None, name='output')
            prob = tf.nn.sigmoid(logits)

        return tf.squeeze(logits, axis=1), prob

    def concat_block(self):
        # Initialze neural network
        with tf.variable_scope("concat_block") as scope:

            concat_layer = tf.concat([self.o1, self.o2], 1)
            fc1 = tf.layers.dense(concat_layer, 1024, activation=tf.nn.relu, name='fc1')
            logits = tf.layers.dense(fc1, 1, activation=None, name='output')
            prob = tf.nn.sigmoid(logits)

        return tf.squeeze(logits, axis=1), prob

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return eucd, losses, tf.nn.sigmoid(losses), loss

    def loss_with_step(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_binary_cross_entr(self):
        # Define loss function
        with tf.variable_scope("loss_function") as scope:
            labels = self.y_
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels, logits=self.pair_logit, name='binary_cross_entr_loss')
            )

        return loss