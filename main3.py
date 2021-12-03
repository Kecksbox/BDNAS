import copy
import datetime
import random
import time

import tensorflow as tf

from sparse_row_matrix import SparseRowMatrix

random.seed(0)
tf.random.set_seed(0)
tf.get_logger().setLevel('ERROR')

min_conn_param = -100.0
max_conn_param = 100.0


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
def special_sub_action(a, b):
    def grad(upstream):
        return upstream, tf.zeros(shape=b.shape)

    y = a - b
    return y, grad


@tf.custom_gradient
def special_ceil_action(x):
    def grad(upstream):
        return upstream

    y = tf.math.ceil(x)
    return y, grad


def binominal_sample_operation(probs):
    sample = special_ceil_action(special_sub_action(probs, tf.random.uniform(tf.shape(probs))))

    return sample


@tf.custom_gradient
def pass_through_sigmoid(x):
    def grad(upstream):
        return upstream

    y = tf.math.sigmoid(x)
    return y, grad


def binary_gate_sigmoid(params, alive_vector):
    tmp = tf.reshape(params.value, shape=(params.value.shape[0], -1))
    tmp = tf.math.sigmoid(tmp)
    tmp2 = SparseRowMatrix(dense_shape=[params.dense_shape[0], tmp.shape[1]])
    tmp2.indices = params.indices
    tmp2.value = tf.cast(tmp, dtype=tf.float32)
    probs = tmp2

    sample = probs.operation(binominal_sample_operation)
    sample = sample.mul_dense(tf.expand_dims(tf.sparse.to_dense(alive_vector), axis=-1))
    return sample, probs


def binary_gate_softmax(params, num_choices: int, alive_vector):
    tmp = tf.reshape(params.value, shape=(params.value.shape[0], -1, num_choices))
    tmp = tf.math.softmax(tmp, axis=-1)
    probs = tmp

    sample = categorical_sample_operation(probs)
    sparse_sample_result = SparseRowMatrix(dense_shape=params.dense_shape)
    sparse_sample_result.value = sample
    sparse_sample_result.indices = list.copy(params.indices)
    expanded_alive_vector = tf.expand_dims(tf.expand_dims(alive_vector, axis=-1), axis=-1)
    sparse_sample_result.mul_dense(expanded_alive_vector)
    return sparse_sample_result, params


def splice_1_operation(a):
    num_on_rows = a.value.shape[0]
    if num_on_rows > 0:
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
    else:
        result = tf.SparseTensor(indices=tf.constant(0, shape=[0, 2], dtype=tf.int64),
                                 values=tf.constant(0, shape=[0, ], dtype=tf.float32),
                                 dense_shape=a.dense_shape)

    return result


@tf.RegisterGradient("SparseReduceMaxSparse")
def _OpNameGrad(a, b, c, d):
    # define your gradient here
    test = tf.expand_dims(tf.sparse.to_dense(tf.SparseTensor(a.outputs[0], c, a.outputs[-1])), axis=0)
    test2 = tf.SparseTensor(a.inputs[0], a.inputs[1], a.inputs[2])
    res = test2 * test
    return tf.constant(0.0, shape=a.inputs[0].shape), res._values, tf.constant(0.0,
                                                                               shape=a.inputs[2].shape), tf.constant(
        0.0, shape=a.inputs[3].shape)


def reduce_connection_mask_matrix_operation(a, num_input_nodes):
    num_output_nodes = a.shape[1] - num_input_nodes
    # result = tf.sparse.reduce_max(a, axis=0, output_is_sparse=True)
    # result = tf.sparse.slice(result, [0], [num_output_nodes])
    #result = tf.reduce_max(tf.sparse.to_dense(a), axis=0)[:num_output_nodes]
    #result = tf.sparse.from_dense(result)
    result = tf.reduce_max(a, axis=0)[:num_output_nodes]
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
]


def init_activation_parameter():
    return tf.constant(0.0, shape=(1, len(activation_function_catalog)), dtype=tf.float32)


def init_bias_vector():
    return tf.random.normal(shape=(1, 1), dtype=tf.float32)


def mul_by_alive_vector(target, alive_vector, init_function):
    dense_alive_vector = alive_vector
    rows_to_init = tf.cast(tf.cast(tf.logical_not(target.indices), dtype=tf.float32) * dense_alive_vector,
                           dtype=tf.bool)
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
    new_target = tf.sparse.add(b, target)
    target = new_target
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

        # self.connection_parameter = SparseRowMatrix(dense_shape=[dim_out, self.num_valid_input_nodes], dtype=tf.float64)
        self.connection_parameter = tf.Variable(tf.constant(0.0, shape=[dim_out, self.num_valid_input_nodes]))

        # self.weight_matrix = tf.SparseTensor(indices=tf.constant(0, shape=[0, 2], dtype=tf.int64),
        #                                     values=tf.constant(0, shape=[0, ], dtype=tf.float32),
        #                                     dense_shape=[dim_out, self.num_valid_input_nodes])

        self.weight_matrix = tf.Variable(tf.random.normal(shape=[dim_out, self.num_valid_input_nodes]))

        self.bias_vector = tf.Variable(tf.random.normal(shape=(dim_out, 1)))

        self.activation_parameter = tf.Variable(tf.constant(0.0, shape=(dim_out, len(activation_function_catalog))))
        self.activation_parameter = SparseRowMatrix(dense_shape=[dim_out, len(activation_function_catalog)])

    def get_weight_variables(self):
        return [self.weight_matrix]

    def get_topologie_variables(self):
        return [self.connection_parameter, self.activation_parameter.value]

    def init_connection_parameter(self):
        return tf.constant(0.0, shape=(1, self.num_valid_input_nodes), dtype=tf.float64)

    def init_weight_parameter(self):
        return tf.random.normal(shape=(self.num_valid_input_nodes,))

    def sample_topologie(self, output_nodes_alive: SparseRowMatrix):
        #connection_parameter = mul_by_alive_vector(self.connection_parameter, output_nodes_alive,
        #                                           init_function=self.init_connection_parameter)
        #self.connection_parameter.value.assign(
        #    tf.clip_by_value(self.connection_parameter.value, min_conn_param, max_conn_param))

        connection_probs = tf.math.sigmoid(self.connection_parameter)
        connection_mask_matrix = binominal_sample_operation(connection_probs)

        #connection_mask_matrix, _ = binary_gate_sigmoid(connection_parameter, output_nodes_alive)

        # connection_mask_matrix = splice_1_operation(connection_mask_matrix)
        #connection_mask_matrix = connection_mask_matrix.to_dense(0.0)

        self.weight_matrix, weight_matrix = mul_by_connection_mask(self.weight_matrix, connection_mask_matrix)

        weight_matrix = connection_mask_matrix * tf.expand_dims(output_nodes_alive, axis=-1) * self.weight_matrix

        # weight_matrix = self.weight_matrix * tf.sparse.to_dense(connection_mask_matrix)

        activation_parameter = mul_by_alive_vector(self.activation_parameter, output_nodes_alive,
                                                   init_function=init_activation_parameter)
        activation_mask_matrix, _ = binary_gate_softmax(activation_parameter, len(activation_function_catalog),
                                                        output_nodes_alive)
        # tf.print("create activation_mask_matrix took {}".format(end - start))

        # bias_vector = mul_by_alive_vector(self.bias_vector, output_nodes_alive,
        #                                  init_function=init_bias_vector)
        bias_vector = self.bias_vector

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

    def sample_topologie(self, max_depth):
        sequence = []
        current_layer = self.output_layer
        output_nodes_alive = tf.constant(1.0, shape=[current_layer.dim_out], dtype=tf.float32)
        depth = 0
        while True:
            weight_matrix, bias_vector, connection_mask_matrix, activation_mask_matrix = current_layer.sample_topologie(
                output_nodes_alive)

            sequence.append(
                (weight_matrix, bias_vector, activation_mask_matrix)
            )

            connection_mask_matrix_reduced_to_nodes = reduce_connection_mask_matrix_operation(connection_mask_matrix,
                                                                                              self.input_layer.dim_in)

            if depth < max_depth and tf.reduce_any(tf.cast(connection_mask_matrix_reduced_to_nodes, dtype=tf.bool)):
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
        num_activations = activation_mask_matrix.dense_shape[-1]
        activations = [None] * num_activations
        for activation_id in range(num_activations):
            local_activation = activation_function_catalog[activation_id](activation)
            activations[activation_id] = tf.expand_dims(local_activation, axis=-1)
        activations = activation_mask_matrix.mul_dense(tf.concat(activations, axis=-1))
        activations.value = tf.reduce_sum(activations.value, axis=-1)
        return activations.to_dense(0.0)

    def __call__(self, input, sequence, *args, **kwargs):

        num_input_nodes = self.input_layer.dim_in
        assert num_input_nodes == input.shape[1]

        first_hidden_layer_weights = sequence[-1][0]
        input = tf.transpose(input)
        activation = tf.constant(0.0, shape=(first_hidden_layer_weights.shape[1] - num_input_nodes, input.shape[1]))

        i = 1
        for weight_matrix, bias_vector, activation_mask_matrix in reversed(sequence):
            activation = tf.concat([activation, input], axis=0)
            activation = tf.matmul(weight_matrix, activation)
            if i < len(sequence):
                i += 1
                activation = tf.keras.activations.relu(activation)
            # activation = bias_vector + activation
            # activation = self.apply_activation_function_catalog(activation, activation_mask_matrix)

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


test = Network(3, 2, 2.4)
opt_weights = tf.keras.optimizers.Adam(0.1)
opt_topo = tf.keras.optimizers.Adam(0.1)

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
    [2.4, 2.3],
    [0.1, 0.11],
    [0.2, 2.31],
    [0.6, 0.11],
    [0.6, 2.41],
])

epoch = 0
while True:
    train_batch_x = test_input
    train_batch_y = test_output

    test_batch_x = test_input
    test_batch_y = test_output

    with tf.GradientTape() as tape:

        sequence = test.sample_topologie(max_depth=3)

        result_train = test(train_batch_x, sequence)

        loss_train = tf.reduce_mean(tf.losses.mean_squared_error(train_batch_y, result_train))

        vars = test.get_topologie_variables() + test.get_weight_variables()
        grads = tape.gradient(loss_train, vars)
        types = [None] * len(grads)
        conv_grads = [None] * len(grads)
        for grad_index in range(len(grads)):
            grad = grads[grad_index]
            if grad is None:
                continue
            types[grad_index] = grad.dtype
            conv_grads[grad_index] = tf.cast(grad, dtype=tf.float64)
        grads, _ = tf.clip_by_global_norm(conv_grads, 1.0)
        for grad_index in range(len(grads)):
            type = types[grad_index]
            if type is None:
                continue
            grads[grad_index] = tf.cast(grads[grad_index], dtype=type)
        try:
            opt_weights.apply_gradients(zip(grads, vars))
        except ValueError:
            pass

    train_loss(loss_train)

    with train_summary_writer.as_default():
        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
    train_loss.reset_states()
    test_loss.reset_states()

    num_active_neurons = 0
    num_active_connections = 0
    for w, b, a in sequence:
        num_active_neurons += 0
        num_active_connections += w.shape[0]

    active_neurons(num_active_neurons)
    active_connections(num_active_connections)
    with train_summary_writer.as_default():
        tf.summary.scalar('active_neurons', active_neurons.result(), step=epoch)
    with train_summary_writer.as_default():
        tf.summary.scalar('active_connections', active_connections.result(), step=epoch)
    active_neurons.reset_states()
    active_connections.reset_states()

    epoch += 1

    top_vars = test.get_topologie_variables()
    tf.print(tf.math.softmax(top_vars[1], axis=-1))
    tf.print(top_vars[0])
    # for k in range(int(len(top_vars))):
    #    tf.print(top_vars[int(k)][0])

    # tf.print(result_train)
    # tf.print(vars[0])
    # tf.print(test.output_layer.activation_parameter[0])
    # tf.print(test.output_layer.activation_parameter[-1])
