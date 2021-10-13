import copy
import datetime
import pickle
import random
import time
from pathlib import Path

import tensorflow as tf

from sparse_row_matrix import SparseRowMatrix

random.seed(0)
tf.random.set_seed(0)
tf.get_logger().setLevel('ERROR')

min_conn_param = -6.0
max_conn_param = 6.0


@tf.custom_gradient
def categorical_sample_operation(probs):
    original_matrix_shape = probs.shape
    probs_2d = tf.reshape(probs, shape=(-1, original_matrix_shape[-1]))
    sample = tf.squeeze(tf.random.categorical(logits=tf.math.log(probs_2d), num_samples=1))
    sample = tf.one_hot(sample, depth=probs_2d.shape[-1])  # on_value=True, off_value=False, dtype=tf.bool)
    sample = tf.reshape(sample, shape=original_matrix_shape)

    def grad(upstream):
        #test = tf.reduce_max(upstream)
        return upstream

    return sample, grad


@tf.custom_gradient
def binominal_sample_operation(probs):
    sample = tf.cast(tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs)))), dtype=tf.float32)

    def grad(upstream):
        #test = tf.reduce_max(upstream)
        return upstream

    return sample, grad


def binary_gate_sigmoid(params, alive_vector, alive_probs_vector, train):
    tmp = tf.reshape(params, shape=(params.shape[0], -1))
    tmp = tf.math.sigmoid(tmp)
    probs = tmp * tf.cast(tf.expand_dims(alive_vector, axis=-1), dtype=tf.float32)

    if train:
        sample = binominal_sample_operation(probs)
    else:
        sample = tf.cast(tf.nn.relu(tf.sign(probs - tf.constant(1 / 2, shape=probs.shape))), dtype=tf.float32)

    return sample, probs * tf.expand_dims(alive_probs_vector, axis=-1)


@tf.custom_gradient
def check(a):
    def grad(upstream):
        test = tf.reduce_max(upstream)
        return upstream

    return a, grad


def binary_gate_softmax(params, num_choices: int, alive_vector, train):
    tmp = tf.reshape(params, shape=(params.shape[0], -1, num_choices))
    tmp = tf.math.softmax(tmp, axis=-1)
    probs = tmp * tf.expand_dims(tf.expand_dims(tf.cast(alive_vector, tf.float32), axis=-1), axis=-1)

    if train:
        sample = categorical_sample_operation(probs)
    else:
        original_matrix_shape = probs.shape
        probs_2d = tf.reshape(probs, shape=(-1, original_matrix_shape[-1]))
        sample = tf.squeeze(tf.argmax(probs_2d, axis=-1))
        sample = tf.one_hot(sample, depth=probs_2d.shape[-1])  # on_value=True, off_value=False, dtype=tf.bool)
        sample = tf.reshape(sample, shape=original_matrix_shape)

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

    a = tf.cast(a, dtype=tf.float32)
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
    return target, target * alive_matrix


variance_scaling = tf.keras.initializers.VarianceScaling()


class Layer:
    def __init__(self, dim_in: int, dim_out: int, branching_factor: float = 2.0):
        self.dim_out = dim_out
        self.num_valid_input_nodes = dim_in + int(dim_out * branching_factor)

        self.connection_parameter = tf.Variable(tf.constant(min_conn_param, shape=(dim_out, self.num_valid_input_nodes)))

        self.weight_matrix = tf.Variable(variance_scaling(shape=(dim_out, self.num_valid_input_nodes)))

        self.bias_vector = tf.Variable(variance_scaling(shape=(dim_out, 1)))

        self.activation_parameter = tf.Variable(tf.constant(0.0, shape=(dim_out, len(activation_function_catalog))))

    def get_weight_variables(self):
        return [self.weight_matrix, self.bias_vector]

    def get_topologie_variables(self):
        return [self.connection_parameter, self.activation_parameter]

    def init_connection_parameter(self):
        return tf.constant(-2.0, shape=(1, self.num_valid_input_nodes))

    def init_weight_parameter(self):
        return tf.Variable(tf.random.normal(shape=(self.num_valid_input_nodes,)))

    def sample_topologie_probs(self, output_nodes_alive_prob: SparseRowMatrix):
        _, connection_prob_matrix = binary_gate(self.connection_parameter, True)
        connection_prob_matrix = connection_prob_matrix.__mul__(output_nodes_alive_prob)
        return connection_prob_matrix

    def sample_topologie(self, output_nodes_alive: SparseRowMatrix, output_nodes_alive_prob, train):
        #self.connection_parameter = tf.Variable(
        #    tf.clip_by_value(self.connection_parameter, min_conn_param, max_conn_param))
        connection_mask_matrix, connection_probs_matrix = binary_gate_sigmoid(self.connection_parameter,
                                                                              output_nodes_alive,
                                                                              output_nodes_alive_prob, train)

        self.weight_matrix, weight_matrix = mul_by_connection_mask(self.weight_matrix, connection_mask_matrix)

        activation_mask_matrix, _ = binary_gate_softmax(self.activation_parameter, len(activation_function_catalog),
                                                        output_nodes_alive, train)

        bias_vector = self.bias_vector * tf.cast(tf.expand_dims(output_nodes_alive, axis=-1), dtype=tf.float32)

        return weight_matrix, bias_vector, connection_mask_matrix, connection_probs_matrix, activation_mask_matrix


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
        output_nodes_alive = tf.constant(1.0, shape=[current_layer.dim_out], dtype=tf.float32)
        output_nodes_alive_probs = tf.constant(1.0, shape=[current_layer.dim_out], dtype=tf.float32)
        depth = 0
        while True:
            weight_matrix, bias_vector, connection_mask_matrix, connection_probs_matrix, activation_mask_matrix = current_layer.sample_topologie(
                output_nodes_alive, output_nodes_alive_probs, train)

            sequence.append(
                (weight_matrix, bias_vector, activation_mask_matrix, connection_probs_matrix, output_nodes_alive_probs)
            )

            connection_mask_matrix_reduced_to_nodes = reduce_connection_mask_matrix_operation(connection_mask_matrix,
                                                                                              self.input_layer.dim_in)

            if depth < max_depth and tf.reduce_any(tf.cast(connection_mask_matrix_reduced_to_nodes, dtype=tf.bool)):
                current_layer = self.get_hidden_layer(depth)
                output_nodes_alive = connection_mask_matrix_reduced_to_nodes

                a = 1 - tf.cast(connection_probs_matrix, dtype=tf.float32)
                num_input_nodes = self.input_layer.dim_in
                num_output_nodes = a.shape[1] - num_input_nodes

                result = 1 - tf.reduce_prod(a[:, :num_output_nodes], axis=0)
                output_nodes_alive_probs = tf.cast(result, dtype=tf.float32)

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

    def decay(self, sequence, limit: float):
        layers = [self.output_layer] + self.hidden_layers
        for i in range(len(sequence)):
            _, _, _, connection_probs_matrix, output_nodes_alive_probs = sequence[i]
            decay_mask = tf.cast(tf.less(connection_probs_matrix, limit), dtype=tf.float32)
            layer = layers[i]
            layer.connection_parameter.assign(
                layer.connection_parameter * tf.cast(tf.logical_not(tf.cast(decay_mask, dtype=tf.bool)),
                                                     dtype=tf.float32) + (
                        decay_mask * tf.constant(min_conn_param, shape=(layer.connection_parameter.shape))))
            decay_mask = tf.cast(tf.greater_equal(output_nodes_alive_probs, limit), dtype=tf.float32)
            layer.activation_parameter.assign(layer.activation_parameter * tf.expand_dims(decay_mask, axis=-1))

    def __call__(self, input, sequence, *args, **kwargs):
        num_input_nodes = self.input_layer.dim_in
        assert num_input_nodes == input.shape[1]

        first_hidden_layer_weights = sequence[-1][0]
        input = tf.transpose(input)
        activation = tf.constant(0.0, shape=(first_hidden_layer_weights.shape[1] - num_input_nodes, input.shape[1]))

        # i = 1
        for weight_matrix, bias_vector, activation_mask_matrix, _, _ in reversed(sequence):
            activation = tf.concat([activation, input], axis=0)
            activation = tf.matmul(weight_matrix, activation)
            activation = bias_vector + activation
            # if i < len(sequence):
            #    activation = tf.keras.activations.relu(activation)
            #    i += 1
            activation = self.apply_activation_function_catalog(activation, activation_mask_matrix)

        return tf.transpose(activation)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)
        tf.print("Saved")

    @staticmethod
    def load(path):
        try:
            with open(path, 'rb') as fp:
                network = pickle.load(fp)
                tf.print("Load")
                return network
        except IOError:
            tf.print("Load Error")
            return None


checkpoint_path = "./checkpoints/check.json"
test = Network.load(checkpoint_path)
max_depth = 3
if test is None:
    test = Network(784, 10, 2.2)
opt_weights = tf.keras.optimizers.SGD(learning_rate=0.01)
opt_topo = tf.keras.optimizers.Adam(learning_rate=0.01)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
topology_loss = tf.keras.metrics.Mean('topology_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
active_neurons = tf.keras.metrics.Mean('active_neurons', dtype=tf.float32)
active_connections = tf.keras.metrics.Mean('active_connections', dtype=tf.float32)

mnist = tf.keras.datasets.mnist
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.cast(tf.reshape(x_train, shape=(-1, x_train.shape[-1] * x_train.shape[-2])), dtype=tf.float32)
x_test = tf.cast(tf.reshape(x_test, shape=(-1, x_test.shape[-1] * x_test.shape[-2])), dtype=tf.float32)

x_test_split = tf.reshape(x_test, shape=(2, -1, x_test.shape[-1]))
y_test_split = tf.reshape(y_test, shape=(2, -1))
y_topology = y_test_split[0]
y_test = y_test_split[1]
x_topology = x_test_split[0, :, :]
x_test = x_test_split[1, :, :]

batches_train = 10
batch_size_train = 200

batches_topology = 5
batch_size_topology = 200

batches_test = 10
batch_size_test = 200

train_dataset = []
for i in range(batches_train):
    train_batch = random.choices(range(0, x_train.shape[0]), k=batch_size_train)
    train_batch_x = tf.stack([x_train[i] for i in train_batch])
    train_batch_y = tf.stack([y_train[i] for i in train_batch])

    train_dataset.append((train_batch_x, train_batch_y))

topology_dataset = []
for i in range(batches_topology):
    topology_batch = random.choices(range(0, x_topology.shape[0]), k=batch_size_topology)
    topology_batch_x = tf.stack([x_topology[i] for i in topology_batch])
    topology_batch_y = tf.stack([y_topology[i] for i in topology_batch])

    topology_dataset.append((topology_batch_x, topology_batch_y))

test_dataset = []
for i in range(batches_test):
    test_batch = random.choices(range(0, x_test.shape[0]), k=batch_size_test)
    test_batch_x = tf.stack([x_test[i] for i in test_batch])
    test_batch_y = tf.stack([y_test[i] for i in test_batch])

    test_dataset.append((test_batch_x, test_batch_y))

train_set = [(False, train_batch_x, train_batch_y) for train_batch_x, train_batch_y in train_dataset] + [
    (True, topology_batch_x, topology_batch_y) for topology_batch_x, topology_batch_y in topology_dataset]
epoch = 0
while True:
    random.shuffle(train_set)
    for b_topology, batch_x, batch_y in train_set:
        with tf.GradientTape() as tape:

            sequence = test.sample_topologie(max_depth=max_depth, train=True)

            result_train = tf.math.softmax(test(batch_x, sequence), axis=-1)

            loss_train = tf.reduce_mean(loss_object(batch_y, result_train))

            if b_topology:
                vars = test.get_topologie_variables()
                grads = tape.gradient(loss_train, vars)
                try:
                    opt_topo.apply_gradients(zip(grads, vars))
                except ValueError:
                    pass

            else:
                vars = test.get_weight_variables()
                grads = tape.gradient(loss_train, vars)
                try:
                    opt_weights.apply_gradients(zip(grads, vars))
                except ValueError:
                    pass

            if b_topology:
                topology_loss(loss_train)

                num_active_neurons = 0
                num_active_connections = 0
                for w, b, a, _, _ in sequence:
                    num_active_neurons += tf.where(a).shape[0]
                    num_active_connections += tf.where(w).shape[0]

                active_neurons(num_active_neurons)
                active_connections(num_active_connections)
            else:
                train_loss(loss_train)

        if b_topology:
            #test.decay(sequence, limit=0.0000001)
            pass

    for test_batch_x, test_batch_y in test_dataset:
        start = time.time()
        sequence = test.sample_topologie(max_depth=max_depth, train=True)
        end = time.time()

        result_test = tf.math.softmax(test(test_batch_x, sequence), axis=-1)
        loss_test = tf.reduce_mean(loss_object(test_batch_y, result_test))

        test_loss(loss_test)

    with train_summary_writer.as_default():
        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
    with train_summary_writer.as_default():
        tf.summary.scalar('topology_loss', topology_loss.result(), step=epoch)
    with train_summary_writer.as_default():
        tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
    train_loss.reset_states()
    topology_loss.reset_states()
    test_loss.reset_states()

    with train_summary_writer.as_default():
        tf.summary.scalar('active_neurons', active_neurons.result(), step=epoch)
    with train_summary_writer.as_default():
        tf.summary.scalar('active_connections', active_connections.result(), step=epoch)
    active_neurons.reset_states()
    active_connections.reset_states()

    epoch += 1

    # tf.print(result_train)
    top_vars = test.get_topologie_variables()
    # tf.print(tf.reduce_max(top_vars[0]))
    tf.print(test.output_layer.activation_parameter[0])
    for k in range(int(len(top_vars) / 2)):
        tf.print(tf.reduce_mean(top_vars[int(k * 2)]))
    # tf.print(test.output_layer.activation_parameter[-1])

    if epoch % 1000 == 0:
        test.save(checkpoint_path)
        test = Network.load(checkpoint_path)
"""

epoch = 0
while True:
    for train_batch_x, train_batch_y, test_batch_x, test_batch_y in dataset:
        with tf.GradientTape(persistent=True) as tape:

            start = time.time()
            sequence = test.sample_topologie(max_depth=3, train=False)
            end = time.time()
            # tf.print("sequence generation took {}".format(end - start))

            result_train = tf.math.softmax(test(train_batch_x, sequence), axis=-1)
            result_test = tf.math.softmax(test(test_batch_x, sequence), axis=-1)

            loss_train = tf.reduce_mean(loss_object(train_batch_y, result_train))
            loss_test = tf.reduce_mean(loss_object(test_batch_y, result_test))

            vars = test.get_weight_variables()
            grads = tape.gradient(loss_train, vars)
            try:
                opt_weights.apply_gradients(zip(grads, vars))
            except ValueError:
                pass

        tape.__del__()

        train_loss(loss_train)
        test_loss(loss_test)

        num_active_neurons = 0
        num_active_connections = 0
        for w, b, a, _, _ in sequence:
            num_active_neurons += tf.where(a).shape[0]
            num_active_connections += tf.where(w).shape[0]

        active_neurons(num_active_neurons)
        active_connections(num_active_connections)

    with train_summary_writer.as_default():
        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
    with train_summary_writer.as_default():
        tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
    train_loss.reset_states()
    test_loss.reset_states()


    with train_summary_writer.as_default():
        tf.summary.scalar('active_neurons', active_neurons.result(), step=epoch)
    with train_summary_writer.as_default():
        tf.summary.scalar('active_connections', active_connections.result(), step=epoch)
    active_neurons.reset_states()
    active_connections.reset_states()

    epoch += 1

    #tf.print(result_train)
    tf.print(tf.reduce_max(vars[0]))
    tf.print(test.output_layer.activation_parameter[0])
    #tf.print(test.output_layer.activation_parameter[-1])
"""
