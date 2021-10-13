import copy
import datetime
import random
import time

import tensorflow as tf

from sparse_row_matrix import SparseRowMatrix

tf.random.set_seed(0)
tf.get_logger().setLevel('ERROR')


@tf.custom_gradient
def categorical_sample_operation(probs):
    original_matrix_shape = probs.shape
    probs_2d = tf.reshape(probs, shape=(-1, original_matrix_shape[-1]))
    sample = tf.squeeze(tf.random.categorical(logits=tf.math.log(probs_2d), num_samples=1))
    sample = tf.one_hot(sample, depth=probs_2d.shape[-1])  # on_value=True, off_value=False, dtype=tf.bool)
    sample = tf.reshape(sample, shape=original_matrix_shape)

    def grad(upstream):
        return upstream

    return sample, grad


@tf.custom_gradient
def binominal_sample_operation(probs):
    sample = tf.cast(tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs)))), dtype=tf.float32)

    def grad(upstream):
        return upstream

    return sample, grad


def binary_gate_sigmoid(params, alive_vector):
    tmp = tf.reshape(params.value, shape=(params.value.shape[0], -1))
    tmp = tf.math.sigmoid(tmp)
    tmp2 = SparseRowMatrix(dense_shape=(params.dense_shape[0], tmp.shape[1]))
    tmp2.indices = params.indices
    tmp2.value = tmp
    probs = tmp2.mul_dense(tf.expand_dims(tf.cast(alive_vector, tf.float32), axis=-1))

    sample = probs.operation(binominal_sample_operation)
    return sample, probs


def binary_gate_softmax(params, num_choices: int, alive_vector):
    tmp = tf.reshape(params, shape=(params.shape[0], -1, num_choices))
    tmp = tf.math.softmax(tmp, axis=-1)
    probs = tmp * tf.expand_dims(tf.expand_dims(tf.cast(alive_vector, tf.float32), axis=-1), axis=-1)

    sample = categorical_sample_operation(probs)
    return sample, params


def splice_1_operation(a):
    # a.value = a.value[:, :, 1]
    # a.dense_shape = (a.dense_shape[0], a.dense_shape[1])

    num_on_rows = a.value.shape[0]
    offsets = [None] * num_on_rows
    indices = tf.where(a.value)
    count = tf.reduce_sum(a.value, axis=-1)
    i = 0
    j = 0
    for index in a.indices:
        if not index:
            i += 1
        else:
            t2 = tf.tile(tf.constant([[i, 0]], dtype=tf.int64), multiples=(count[j], 1))
            offsets[j] = t2
            j += 1

    indices = indices + tf.concat(offsets, axis=0)
    values = tf.boolean_mask(a.value, a.value)
    result = tf.SparseTensor(indices=indices,
                             values=values,
                             dense_shape=a.dense_shape)

    return result


def reduce_connection_mask_matrix_operation(a, num_input_nodes):
    num_output_nodes = a.shape[1] - num_input_nodes

    a._values = tf.cast(a._values, dtype=tf.uint8)
    result = tf.cast(tf.sparse.reduce_max(a, axis=0), dtype=tf.bool)[:num_output_nodes]
    return result


def reduce_any_operation(a):
    return tf.reduce_any(tf.cast(tf.stack(a.rows), dtype=tf.bool))


def arg_max_operation(a):
    arg_max = tf.argmax(tf.stack(a.rows), axis=-1)
    probs = tf.one_hot(arg_max, depth=a.shape[-1])
    return SparseRowMatrix(shape=a.shape, rows=tf.unstack(probs), row_keys=a.keys)


def splice_0_reduce_prod_operation(a, num_input_nodes):
    num_output_nodes = a.shape[1] - num_input_nodes
    if a.is_empty():
        return SparseRowMatrix(shape=(num_output_nodes, 1), rows=[], row_keys=[])

    A = tf.stack(a.rows)
    result = A[:, :, 0]
    result = 1 - tf.expand_dims(tf.expand_dims(tf.reduce_prod(result, axis=0)[:num_output_nodes], axis=-1), axis=-1)
    return SparseRowMatrix(shape=(num_output_nodes, 1, 1), rows=tf.unstack(result),
                           row_keys=list(range(num_output_nodes)))


def limit_mask_operation(a, limit: float):
    A = tf.stack(a.rows)
    result = tf.cast(tf.greater_equal(A, limit), tf.float32)
    return SparseRowMatrix(shape=a.shape, rows=tf.unstack(result),
                           row_keys=a.keys)


class InputLayer:
    def __init__(self, dim_in: int):
        self.input = tf.Variable(0)
        self.dim_in = dim_in

    def assign(self, input):
        self.input.assign(input)


activation_function_catalog = [
    tf.keras.activations.linear,
    tf.keras.activations.relu,
    tf.keras.activations.sigmoid,
    tf.keras.activations.tanh,
]


def init_activation_parameter():
    return tf.constant(0.0, shape=(1, len(activation_function_catalog)), dtype=tf.float32)


def init_bias_vector():
    return tf.random.normal(shape=(1, 1), dtype=tf.float32)


def mul_by_alive_vector(target, alive_vector, init_function):
    rows_to_init = tf.logical_and(tf.logical_not(target.indices), alive_vector)
    for index in tf.where(rows_to_init):
        target.assign(init_function(), tf.squeeze(index))
    target.value = tf.Variable(target.value)
    return target


def mul_by_connection_mask(target, alive_matrix):
    old_alive_matrix_values = alive_matrix._values
    alive_matrix._values = tf.random.normal(shape=old_alive_matrix_values.shape)
    old_target_values = target._values
    target._values = tf.constant(1.0, shape=old_target_values.shape)
    a = alive_matrix * tf.sparse.to_dense(target)
    a._values *= -1
    b = tf.sparse.add(a, alive_matrix)
    target._values = old_target_values
    target = tf.sparse.add(b, target)
    target._values = tf.Variable(tf.stop_gradient(target._values))
    alive_matrix._values = old_alive_matrix_values

    target_alive = alive_matrix * tf.sparse.to_dense(target)
    b_mask = tf.cast(target_alive._values, dtype=tf.bool)
    target_alive._indices = tf.boolean_mask(target_alive._indices, b_mask)
    target_alive._values = tf.boolean_mask(target_alive._values, b_mask)

    return target, target_alive


class Layer:
    def __init__(self, dim_in: int, dim_out: int, branching_factor: float = 2.0):
        self.dim_out = dim_out
        self.num_valid_input_nodes = dim_in + int(dim_out * branching_factor)

        self.connection_parameter = SparseRowMatrix(dense_shape=[dim_out, self.num_valid_input_nodes])

        self.weight_matrix = tf.SparseTensor(indices=tf.constant(0, shape=[0, 2], dtype=tf.int64),
                                             values=tf.constant(0, shape=[0, ], dtype=tf.float32),
                                             dense_shape=[dim_out, self.num_valid_input_nodes])

        self.bias_vector = tf.Variable(tf.random.normal(shape=(dim_out, 1)))

        self.activation_parameter = tf.Variable(tf.constant(0.0, shape=(dim_out, len(activation_function_catalog))))
        # self.activation_parameter = SparseRowMatrix(shape=(dim_out, len(activation_function_catalog)), rows=[],
        #                                            row_keys=[])
        # self.weight_matrix = SparseRowMatrix(shape=(dim_out, self.num_valid_input_nodes), rows=[], row_keys=[])

    def get_weight_variables(self):
        return [self.weight_matrix._values]

    def get_topologie_variables(self):
        return [self.connection_parameter.value]

    def init_connection_parameter(self):
        return tf.constant(-2.0, shape=(1, self.num_valid_input_nodes))

    def init_weight_parameter(self):
        return tf.Variable(tf.random.normal(shape=(self.num_valid_input_nodes,)))

    def sample_topologie_probs(self, output_nodes_alive_prob: SparseRowMatrix):
        _, connection_prob_matrix = binary_gate(self.connection_parameter, True)
        connection_prob_matrix = connection_prob_matrix.__mul__(output_nodes_alive_prob)
        return connection_prob_matrix

    def sample_topologie(self, output_nodes_alive: SparseRowMatrix, train):
        start = time.time()
        connection_parameter = mul_by_alive_vector(self.connection_parameter, output_nodes_alive,
                                                   init_function=self.init_connection_parameter)
        connection_parameter.value.assign(tf.clip_by_value(connection_parameter.value, -4.0, 4.0))

        end = time.time()
        # tf.print("create conn param took {}".format(end - start))
        connection_mask_matrix, _ = binary_gate_sigmoid(connection_parameter, output_nodes_alive)

        # connection_mask_matrix.value = controll(connection_mask_matrix.value)

        start = time.time()
        # connection_mask_matrix = connection_mask_matrix.to_dense(0.0)
        # connection_mask_matrix = tf.sparse.to_dense(tf.sparse.from_dense(connection_mask_matrix.to_dense(0.0)))
        connection_mask_matrix = splice_1_operation(connection_mask_matrix)
        end = time.time()
        # tf.print("splice took {}".format(end - start))

        # connection_mask_matrix = controll(connection_mask_matrix)

        start = time.time()
        self.weight_matrix, weight_matrix = mul_by_connection_mask(self.weight_matrix, connection_mask_matrix)
        end = time.time()
        # tf.print("create weight param took {}".format(end - start))

        start = time.time()
        activation_mask_matrix, _ = binary_gate_softmax(self.activation_parameter, len(activation_function_catalog),
                                                        output_nodes_alive)
        end = time.time()
        # tf.print("create activation_mask_matrix took {}".format(end - start))

        # bias_vector = mul_by_alive_vector(self.bias_vector, output_nodes_alive,
        #                                  init_function=init_bias_vector)
        bias_vector = self.bias_vector * tf.cast(tf.expand_dims(output_nodes_alive, axis=-1), dtype=tf.float32)

        return weight_matrix, bias_vector, connection_mask_matrix, activation_mask_matrix


class Network:
    def __init__(self, dim_in, dim_out, branching_factor: float):
        self.branching_factor = branching_factor

        self.input_layer = InputLayer(dim_in)
        self.output_layer = Layer(dim_in, dim_out, self.branching_factor)
        self.hidden_layers = []

    def get_weight_variables(self):
        variables = self.output_layer.get_weight_variables()
        for l in self.hidden_layers:
            variables += l.get_weight_variables()
        return variables

    def get_topologie_variables(self):
        variables = self.output_layer.get_topologie_variables()
        for l in self.hidden_layers:
            variables += l.get_topologie_variables()
        return variables

    def get_hidden_layer(self, depth: int):
        if depth < 0:
            return self.output_layer
        if len(self.hidden_layers) <= depth:
            next_layer = self.get_hidden_layer(depth - 1)
            self.hidden_layers.append(Layer(
                self.input_layer.dim_in,
                next_layer.num_valid_input_nodes - self.input_layer.dim_in,
                self.branching_factor,
            ))
        return self.hidden_layers[depth]

    def sample_topologie_probs(self, max_depth):
        sequence = []
        current_layer = self.output_layer
        output_nodes_alive_prob = SparseRowMatrix(
            shape=(current_layer.dim_out, 1, 1), rows=[tf.constant(1.0, shape=(1, 1))] * current_layer.dim_out,
            row_keys=list(range(current_layer.dim_out))
        )
        depth = 0
        while True:
            connection_prob_matrix = current_layer.sample_topologie_probs(output_nodes_alive_prob)

            sequence.append(
                (connection_prob_matrix, output_nodes_alive_prob)
            )
            if depth < max_depth:
                current_layer = self.get_hidden_layer(depth)
                output_nodes_alive_prob = splice_0_reduce_prod_operation(connection_prob_matrix,
                                                                         self.input_layer.dim_in)
                depth += 1
            else:
                break
        return sequence

    def sample_topologie(self, max_depth, train: bool):
        sequence = []
        current_layer = self.output_layer
        output_nodes_alive = tf.constant(True, shape=[current_layer.dim_out], dtype=tf.bool)
        depth = 0
        while True:
            weight_matrix, bias_vector, connection_mask_matrix, activation_mask_matrix = current_layer.sample_topologie(
                output_nodes_alive, train)

            sequence.append(
                (weight_matrix, bias_vector, activation_mask_matrix)
            )

            connection_mask_matrix_reduced_to_nodes = reduce_connection_mask_matrix_operation(connection_mask_matrix,
                                                                                              self.input_layer.dim_in)

            if depth < max_depth and tf.reduce_any(connection_mask_matrix_reduced_to_nodes):
                current_layer = self.get_hidden_layer(depth)
                output_nodes_alive = connection_mask_matrix_reduced_to_nodes
                depth += 1
            else:
                break
        return sequence

    def append_input(self, activation, input_rows, num_input_nodes):
        activation.rows += input_rows
        activation.keys += list(range(activation.shape[0], activation.shape[0] + num_input_nodes))
        activation.shape = (activation.shape[0] + num_input_nodes, activation.shape[1])

    def apply_activation_function_catalog(self, activation, activation_mask_matrix):
        activations = [None] * activation_mask_matrix.shape[-1]
        for activation_id in range(activation_mask_matrix.shape[-1]):
            activations[activation_id] = tf.expand_dims(
                activation_function_catalog[activation_id](activation), axis=-1)
        activations = tf.concat(activations, axis=-1)
        result = activation_mask_matrix * activations
        result = tf.reduce_sum(result, axis=-1)
        return result

    def __call__(self, input, sequence, *args, **kwargs):

        start = time.time()
        num_input_nodes = self.input_layer.dim_in
        assert num_input_nodes == input.shape[1]

        first_hidden_layer_weights = sequence[-1][0]
        input = tf.transpose(input)
        activation = tf.constant(0.0, shape=(first_hidden_layer_weights.shape[1] - num_input_nodes, input.shape[1]))

        i = 1
        for weight_matrix, bias_vector, activation_mask_matrix in reversed(sequence):
            activation = tf.concat([activation, input], axis=0)
            activation = tf.sparse.sparse_dense_matmul(weight_matrix, activation)
            if i < len(sequence):
                i += 1
                activation = tf.keras.activations.sigmoid(activation)
            # activation = bias_vector + activation
            # activation = self.apply_activation_function_catalog(activation, activation_mask_matrix)

        end = time.time()
        # tf.print("sequence call took {}".format(end - start))

        return tf.transpose(activation)

    def decay(self, limit, max_depth):
        sequence = self.sample_topologie_probs(max_depth)

        depth = len(sequence) - 2
        for connection_prob_matrix, output_nodes_alive_prob in reversed(
                sequence):

            layer = self.get_hidden_layer(depth)
            if output_nodes_alive_prob.is_empty():
                output_nodes_alive_prob = SparseRowMatrix(
                    shape=(output_nodes_alive_prob.shape[0], 1), rows=[], row_keys=[],
                )
            else:
                output_nodes_alive_prob = SparseRowMatrix(
                    shape=(output_nodes_alive_prob.shape[0], 1),
                    rows=tf.unstack(tf.squeeze(tf.stack(output_nodes_alive_prob.rows), axis=-1)),
                    row_keys=output_nodes_alive_prob.keys,
                )
            layer.activation_parameter = layer.activation_parameter * (limit_mask_operation(
                output_nodes_alive_prob, limit))
            for i in range(len(layer.activation_parameter.rows)):
                layer.activation_parameter.rows[i] = tf.Variable(layer.activation_parameter.rows[i])

            if connection_prob_matrix.is_empty():
                connection_prob_matrix = SparseRowMatrix(
                    shape=(connection_prob_matrix.shape[0], connection_prob_matrix.shape[1], 1), rows=[], row_keys=[],
                )
            else:
                connection_prob_matrix = SparseRowMatrix(
                    shape=(connection_prob_matrix.shape[0], connection_prob_matrix.shape[1], 1),
                    rows=tf.unstack(tf.expand_dims(tf.stack(connection_prob_matrix.rows)[:, :, 1], axis=-1)),
                    row_keys=connection_prob_matrix.keys,
                )
            layer.connection_parameter = layer.connection_parameter * limit_mask_operation(
                connection_prob_matrix, limit)
            for i in range(len(layer.connection_parameter.rows)):
                layer.connection_parameter.rows[i] = tf.Variable(layer.connection_parameter.rows[i])

            depth -= 1


"""
test = Network(784, 10, 2.0)
opt = tf.keras.optimizers.Adam(0.01)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('greedy_loss', dtype=tf.float32)

test_input = tf.constant([
    [0.3, 0.1, 0.1],
    [0.1, 0.3, 0.2],
    [0.3, 0.6, 0.3],
    [0.9, 1.1, 0.4],
    [0.3, 0.1, 0.5],
])
test_output = tf.constant([
    [0.3, -0.21],
    [0.1, -0.11],
    [0.2, -0.31],
    [0.21, 0.11],
    [0.6, 0.41],
])

mnist = tf.keras.datasets.mnist
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.cast(tf.reshape(x_train, shape=(-1, x_train.shape[-1] * x_train.shape[-2])), dtype=tf.float32)
x_test = tf.cast(tf.reshape(x_test, shape=(-1, x_test.shape[-1] * x_test.shape[-2])), dtype=tf.float32)

epoch = 0
batch_size = 100
while True:
    train_batch = random.choices(range(0, x_train.shape[0]), k=batch_size)
    train_batch_x = tf.stack([x_train[i] for i in train_batch])
    train_batch_y = tf.stack([y_train[i] for i in train_batch])

    test_batch = random.choices(range(0, x_test.shape[0]), k=batch_size)
    test_batch_x = tf.stack([x_test[i] for i in test_batch])
    test_batch_y = tf.stack([y_test[i] for i in test_batch])

    with tf.GradientTape(persistent=True) as tape:

        start = time.time()
        sequence = test.sample_topologie(max_depth=4, train=True)
        end = time.time()
        tf.print("sequence generation took {}".format(end - start))

        result_train = tf.math.softmax(test(train_batch_x, sequence), axis=-1)
        result_test = tf.math.softmax(test(test_batch_x, sequence), axis=-1)

        loss_train = loss_object(train_batch_y, result_train)
        loss_test = loss_object(test_batch_y, result_test)

        vars = test.get_weight_variables()
        grads = tape.gradient(loss_train, vars)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        try:
            opt.apply_gradients(zip(grads, vars))
        except ValueError:
            pass

        vars = test.get_topologie_variables()
        grads = tape.gradient(loss_test, vars)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        try:
            opt.apply_gradients(zip(grads, vars))
        except ValueError:
            pass

    tape.__del__()

    train_loss(loss_train)
    test_loss(loss_test)

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
    with train_summary_writer.as_default():
        tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
    epoch += 1
    train_loss.reset_states()
    test_loss.reset_states()
    # tf.print(result)
    # tf.print(test.output_layer.activation_parameter.rows[0])
    # tf.print(test.output_layer.activation_parameter.rows[-1])
"""

test = Network(3, 2, 2.0)
opt_weights = tf.keras.optimizers.Adam(0.001)
opt_topo = tf.keras.optimizers.Adam(0.001)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
active_neurons = tf.keras.metrics.Mean('active_neurons', dtype=tf.float32)
active_connections = tf.keras.metrics.Mean('active_connections', dtype=tf.float32)

test_input = tf.constant([
    [0.3, 0.1, 0.1],
    [0.1, 0.3, 0.2],
    [0.3, 0.6, 0.3],
    [0.9, 1.1, 0.4],
    [0.3, 0.1, 0.5],
])
test_output = tf.constant([
    [-0.4, -0.3],
    [0.1, -0.11],
    [0.2, -0.31],
    [-0.21, -0.11],
    [0.6, -0.41],
])

epoch = 0
while True:
    train_batch_x = test_input
    train_batch_y = test_output

    test_batch_x = test_input
    test_batch_y = test_output

    with tf.GradientTape(persistent=True) as tape:

        start = time.time()
        sequence = test.sample_topologie(max_depth=3, train=True)
        end = time.time()
        # tf.print("sequence generation took {}".format(end - start))

        result_train = test(train_batch_x, sequence)
        result_test = test(test_batch_x, sequence)

        loss_train = tf.reduce_mean(tf.losses.mean_squared_error(train_batch_y, result_train))
        loss_test = tf.reduce_mean(tf.losses.mean_squared_error(test_batch_y, result_test))

        vars = test.get_weight_variables()
        grads = tape.gradient(loss_train, vars)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        try:
            opt_weights.apply_gradients(zip(grads, vars))
        except ValueError:
            pass

        vars = test.get_topologie_variables()
        grads = tape.gradient(loss_test, vars)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        try:
            opt_topo.apply_gradients(zip(grads, vars))
        except ValueError:
            pass

    tape.__del__()

    train_loss(loss_train)
    test_loss(loss_test)

    with train_summary_writer.as_default():
        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
    with train_summary_writer.as_default():
        tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
    train_loss.reset_states()
    test_loss.reset_states()

    num_active_neurons = 0
    num_active_connections = 0
    for w, b, a in sequence:
        num_active_neurons += 1
        num_active_connections += 1

    active_neurons(num_active_neurons)
    active_connections(num_active_connections)
    with train_summary_writer.as_default():
        tf.summary.scalar('active_neurons', active_neurons.result(), step=epoch)
    with train_summary_writer.as_default():
        tf.summary.scalar('active_connections', active_connections.result(), step=epoch)
    active_neurons.reset_states()
    active_connections.reset_states()

    epoch += 1

    tf.print(result_train)
    tf.print(vars[0])
    tf.print(test.output_layer.activation_parameter[0])
    tf.print(test.output_layer.activation_parameter[-1])
