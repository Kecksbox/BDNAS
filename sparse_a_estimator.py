import copy
import datetime
import os
import pickle
import random
import time
from pathlib import Path
from typing import List, Callable

import tensorflow as tf

from sparse_row_matrix_v2 import SparseRowMatrix

tf.get_logger().setLevel('ERROR')

min_conn_param = -10.0
max_conn_param = 10.0
inital_virtual_layer_output_node_alive_prob = 0.01
inital_input_layer_node_alive_prob = inital_virtual_layer_output_node_alive_prob

a_estimate_lr = 0.01


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


def binary_gate_sigmoid(params):
    tmp = tf.reshape(tf.stack(params.values), shape=(len(params.values), -1))
    tmp = tf.math.sigmoid(tmp)
    tmp2 = SparseRowMatrix(dense_shape=[params.dense_shape[0], tmp.shape[1]])
    tmp2.indices = params.indices
    tmp2.values = tf.unstack(tmp)
    probs = tmp2

    sample = probs.operation(binominal_sample_operation)
    return sample, probs


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


def binary_gate_softmax(params):
    tmp = tf.reshape(tf.stack(params.values), shape=(len(params.values), -1))
    tmp = tf.math.softmax(tmp, axis=-1)
    tmp2 = SparseRowMatrix(dense_shape=[params.dense_shape[0], tmp.shape[1]])
    tmp2.indices = params.indices
    tmp2.values = tf.unstack(tmp)
    probs = tmp2

    sample = probs.operation(categorical_sample_operation)
    return sample, probs


@tf.custom_gradient
def check(weight_matrix: SparseRowMatrix, activation, previous_actvation_estimate):
    def grad(upstream):
        wm_grad_st = tf.matmul(upstream, tf.transpose(previous_actvation_estimate))
        wm_grad = wm_grad_st
        a_grad = tf.matmul(tf.transpose(weight_matrix), upstream)
        previous_actvation_estimate_grad = tf.zeros(shape=previous_actvation_estimate.shape)
        return wm_grad, a_grad, previous_actvation_estimate_grad

    r = tf.matmul(weight_matrix, activation)

    return r, grad


def reduce_connection_mask_matrix_operation(a, num_input_nodes):
    num_output_nodes = a.dense_shape[1] - num_input_nodes

    tmp = tf.cast(tf.stack(a.values), dtype=tf.bool)[:, :num_output_nodes]
    tmp_reduced = tf.reduce_any(tmp, axis=0)
    return tmp_reduced


def mul_by_alive_vector(target, alive_vector, init_function):
    rows_to_init = tf.logical_and(tf.logical_not(target.indices), alive_vector)
    for index in tf.where(rows_to_init):
        target.assign(init_function(), tf.squeeze(index))

    # remove rows that are dead
    b_alive_masked = tf.boolean_mask(alive_vector, target.indices)
    target_masked = SparseRowMatrix(dense_shape=target.dense_shape)
    target_masked.values = tf.unstack(tf.boolean_mask(tf.stack(target.values), b_alive_masked))
    target_masked.indices = tf.unstack(alive_vector)

    return target, target_masked


class InputLayer:
    def __init__(self, dim_in: int):
        self.input = tf.Variable(0)
        self.dim_in = dim_in

    def assign(self, input):
        self.input.assign(input)


activation_function_catalog = [
    tf.keras.activations.linear,
    tf.keras.activations.relu,
    tf.keras.activations.tanh,
    tf.keras.activations.sigmoid,
]


class Layer:
    def __init__(self, dim_input_layer: int, dim_previous_layer: int, dim_out: int):
        self.dim_input_layer = dim_input_layer
        self.dim_previous_layer = dim_previous_layer
        self.dim_out = dim_out

        self.num_valid_input_nodes = self.dim_input_layer + self.dim_previous_layer

        self.connection_parameter = SparseRowMatrix(dense_shape=[dim_out, self.num_valid_input_nodes])

        self.weight_matrix = SparseRowMatrix(dense_shape=[dim_out, self.num_valid_input_nodes])

        self.layer_norm = tf.keras.layers.LayerNormalization()

        self.activation_parameter = SparseRowMatrix(dense_shape=[dim_out, len(activation_function_catalog)])

    def get_weight_variables(self):
        return self.weight_matrix.values

    def get_topologie_variables(self):
        return self.connection_parameter.values + self.layer_norm.trainable_variables + self.activation_parameter.values

    def init_connection_parameter(self):
        y = inital_virtual_layer_output_node_alive_prob / self.dim_out
        return tf.Variable(tf.constant(tf.math.log(y / (1 - y)), shape=(self.num_valid_input_nodes,)))

    def init_weight_parameter(self):
        return tf.Variable(tf.random.normal(shape=(self.num_valid_input_nodes,)))

    def init_activation_paramter(self):
        return tf.Variable(tf.zeros(shape=(len(activation_function_catalog))))

    def sample_topologie(self, output_nodes_alive: SparseRowMatrix):
        for v in self.connection_parameter.values:
            v.assign(tf.clip_by_value(v, min_conn_param, max_conn_param))

        self.connection_parameter, connection_parameter_masked = mul_by_alive_vector(self.connection_parameter,
                                                                                     output_nodes_alive,
                                                                                     init_function=self.init_connection_parameter)

        connection_mask_matrix, _ = binary_gate_sigmoid(connection_parameter_masked)

        self.weight_matrix, weight_matrix_masked = mul_by_alive_vector(self.weight_matrix,
                                                                       output_nodes_alive,
                                                                       init_function=self.init_weight_parameter)

        weight_matrix = SparseRowMatrix(dense_shape=weight_matrix_masked.dense_shape)
        weight_matrix.values = tf.unstack(
            tf.stack(weight_matrix_masked.values) * tf.stack(connection_mask_matrix.values))
        weight_matrix.indices = weight_matrix_masked.indices

        self.activation_parameter, activation_parameter_masked = mul_by_alive_vector(self.activation_parameter,
                                                                                     output_nodes_alive,
                                                                                     init_function=self.init_activation_paramter)
        activation_mask, _ = binary_gate_softmax(activation_parameter_masked)

        return weight_matrix, connection_mask_matrix, activation_mask, self.layer_norm


def apply_activation(activation, activation_mask):
    activation_map: List[SparseRowMatrix or None] = [None] * len(activation_function_catalog)
    for id in range(len(activation_function_catalog)):
        activation_map[id] = activation.operation(activation_function_catalog[id])

    a = tf.concat([tf.expand_dims(tf.stack(a.values), axis=-1) for a in activation_map], axis=-1)
    b = tf.expand_dims(tf.stack(activation_mask.values), axis=1)
    r = tf.reduce_sum(a * b, axis=-1)
    res = SparseRowMatrix(activation.dense_shape)
    res.values = tf.unstack(r)
    res.indices = list.copy(activation.indices)

    return res


class Network:
    def __init__(self, dim_in, dim_out, virtual_layers: List[Layer]):
        self.input_layer = InputLayer(dim_in)

        dim_previous_layer = 0
        if len(virtual_layers) > 0:
            dim_previous_layer = virtual_layers[0].dim_out
        self.output_layer = Layer(dim_in, dim_previous_layer, dim_out)

        self.hidden_layers = virtual_layers

    def get_weight_variables(self):
        variables = []
        variables += self.output_layer.get_weight_variables()
        for l in self.hidden_layers:
            variables += l.get_weight_variables()
        return variables

    def get_topologie_variables(self):
        variables = []
        variables += self.output_layer.get_topologie_variables()
        for l in self.hidden_layers:
            variables += l.get_topologie_variables()
        return variables

    def get_topologie_variables_grouped(self):
        variables = []
        variables += [self.output_layer.connection_parameter.values]
        for l in self.hidden_layers:
            variables += [l.connection_parameter.values]
        return variables

    def get_hidden_layer(self, depth: int):
        if depth < 0:
            return self.output_layer
        assert depth < len(self.hidden_layers)
        return self.hidden_layers[depth]

    def sample_topologie(self):
        sequence = []
        current_layer = self.output_layer
        output_nodes_alive = tf.constant(True, shape=[current_layer.dim_out], dtype=tf.bool)
        depth = 0
        while True:
            weight_matrix, connection_mask_matrix, activation_mask, layer_norm = current_layer.sample_topologie(
                output_nodes_alive)

            input_nodes_alive = reduce_connection_mask_matrix_operation(connection_mask_matrix, self.input_layer.dim_in)

            sequence.append(
                (
                    weight_matrix, connection_mask_matrix, activation_mask, layer_norm, input_nodes_alive,
                    output_nodes_alive
                )
            )

            if depth < len(self.hidden_layers) and tf.reduce_any(input_nodes_alive):
                current_layer = self.get_hidden_layer(depth)
                output_nodes_alive = input_nodes_alive

                depth += 1
            else:
                break
        return sequence

    def decay(self, limit: float):
        current_layer = self.output_layer
        output_nodes_alive_prob = tf.ones(shape=(current_layer.dim_out,))
        for virtual_layer in [self.output_layer] + self.hidden_layers:
            assert isinstance(virtual_layer, Layer)

            # remove rows by limit
            tf.print(tf.reduce_mean(output_nodes_alive_prob))
            decay_save_mask = tf.greater_equal(output_nodes_alive_prob, limit)
            target = virtual_layer.connection_parameter
            b_alive_mask = tf.boolean_mask(decay_save_mask, target.indices)
            b_dead_mask = tf.logical_not(b_alive_mask)
            for i in reversed(range(b_dead_mask.shape[0])):
                if b_dead_mask[i]:
                    target.values.pop(i)
            target.indices = tf.unstack(tf.logical_and(decay_save_mask, target.indices))

            # calculate probability for next layer
            if len(target.values) > 0:
                alive_probabilities_connections = virtual_layer.connection_parameter.operation(tf.math.sigmoid)
                alive_probabilities_connections_masked = alive_probabilities_connections.mul_dense(
                    tf.expand_dims(output_nodes_alive_prob, axis=-1)
                )

                a = 1 - tf.stack(alive_probabilities_connections_masked.values)
                num_input_nodes = self.input_layer.dim_in
                num_output_nodes = a.shape[1] - num_input_nodes

                result = 1 - tf.reduce_prod(a[:, :num_output_nodes], axis=0)
                output_nodes_alive_prob = result
            else:
                output_nodes_alive_prob = tf.zeros(
                    shape=(virtual_layer.num_valid_input_nodes - self.input_layer.dim_in,)
                )

    def __call__(self, input_batch, sequence, *args, **kwargs):
        sequence_length = len(sequence)
        input_batch, batch_activation_history = input_batch

        num_input_nodes = self.input_layer.dim_in
        assert num_input_nodes == input_batch.shape[1]

        first_hidden_layer_weights = sequence[-1][0]
        input = tf.transpose(input_batch)
        activation = tf.constant(0.0,
                                 shape=(first_hidden_layer_weights.dense_shape[1] - num_input_nodes, input.shape[1]))

        i = 1
        for weight_matrix, connection_mask_matrix, activation_mask, layer_norm, input_nodes_alive, _ in reversed(
                sequence):

            input_nodes_alive = tf.cast(input_nodes_alive, tf.float32)

            history_index = -1 * (i - sequence_length)
            if history_index >= len(batch_activation_history):
                for _ in range(history_index - len(batch_activation_history) + 1):
                    batch_activation_history.append(None)
            if batch_activation_history[history_index] is None:
                batch_activation_history[history_index] = tf.zeros(shape=activation.shape)
            else:
                batch_activation_history[history_index] += (a_estimate_lr * (
                        activation - batch_activation_history[history_index])) * tf.expand_dims(input_nodes_alive,
                                                                                                axis=-1)

            previous_activation = batch_activation_history[history_index] * tf.expand_dims(
                tf.cast(tf.logical_not(tf.cast(input_nodes_alive, dtype=tf.bool)), dtype=tf.float32),
                axis=-1) + activation * tf.expand_dims(input_nodes_alive, axis=-1)

            activation = tf.concat([activation, input], axis=0)

            r_values = tf.unstack(
                check(tf.stack(weight_matrix.values), activation, tf.concat([previous_activation, input], axis=0)))
            r = SparseRowMatrix(dense_shape=[weight_matrix.dense_shape[0], activation.shape[-1]])
            r.values = tf.unstack(r_values)
            r.indices = list.copy(weight_matrix.indices)
            activation = r

            activation = apply_activation(activation, activation_mask)

            if i < len(sequence):
                activation.values = tf.unstack(layer_norm(tf.stack(activation.values)))

            i += 1

            activation = activation.to_dense(0.0, dtype=tf.float32)

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
dim_input = 784
dim_output = 10
if test is None:
    test = Network(dim_input, dim_output, [
        Layer(dim_input, 200, 200),
        Layer(dim_input, 200, 200),
        Layer(dim_input, 200, 200),
        Layer(dim_input, 0, 200),
    ])
opt_topo = tf.keras.optimizers.Adam(learning_rate=0.01)

# work_dir = os.environ["WORK"]
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + "sparse_a_evaluation" + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
topology_loss = tf.keras.metrics.Mean('topology_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
active_neurons = tf.keras.metrics.Mean('active_neurons', dtype=tf.float32)
active_connections = tf.keras.metrics.Mean('active_connections', dtype=tf.float32)
mean_depth = tf.keras.metrics.Mean('mean_depth', dtype=tf.float32)

layers = [test.output_layer] + test.hidden_layers
params_writers = []
for i in range(len(layers)):
    params_writers.append(
        tf.keras.metrics.Mean('connection_param_l{}'.format(i), dtype=tf.float32)
    )

""" Load the DataSet """

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.cast(tf.reshape(x_train, shape=(-1, x_train.shape[-1] * x_train.shape[-2])), dtype=tf.float32)
x_test = tf.cast(tf.reshape(x_test, shape=(-1, x_test.shape[-1] * x_test.shape[-2])), dtype=tf.float32)

x_train_split = tf.reshape(x_train, shape=(2, -1, x_train.shape[-1]))
y_train_split = tf.reshape(y_train, shape=(2, -1))
y_topology = y_train_split[0]
y_train = y_train_split[1]
x_topology = x_train_split[0, :, :]
x_train = x_train_split[1, :, :]

batches_train = 0
batch_size_train = 100

batches_topology = 40
batch_size_topology = 100

batches_test = 5
batch_size_test = 100

train_dataset = []
for i in range(batches_train):
    train_batch = random.choices(range(0, x_train.shape[0]), k=batch_size_train)
    train_batch_x = tf.stack([x_train[i] for i in train_batch])
    train_batch_y = tf.stack([y_train[i] for i in train_batch])

    train_dataset.append(((train_batch_x, []), train_batch_y))

topology_dataset = []
for i in range(batches_topology):
    topology_batch = random.choices(range(0, x_topology.shape[0]), k=batch_size_topology)
    topology_batch_x = tf.stack([x_topology[i] for i in topology_batch])
    topology_batch_y = tf.stack([y_topology[i] for i in topology_batch])

    topology_dataset.append(((topology_batch_x, []), topology_batch_y))

test_dataset = []
for i in range(batches_test):
    test_batch = random.choices(range(0, x_test.shape[0]), k=batch_size_test)
    test_batch_x = tf.stack([x_test[i] for i in test_batch])
    test_batch_y = tf.stack([y_test[i] for i in test_batch])

    test_dataset.append(((test_batch_x, []), test_batch_y))

""" Load the DataSet End """


def apply_batch(batch, loss_function: Callable, bweights: bool, btopology: bool):
    batch_x, batch_y = batch
    with tf.GradientTape(persistent=False) as tape:

        if not (bweights or btopology):
            tape.stop_recording()

        sequence = test.sample_topologie()

        result_train = test(batch_x, sequence)

        loss = loss_function(batch_y, result_train)

        variables = []
        if bweights:
            variables += test.get_weight_variables()
        if btopology:
            variables += test.get_topologie_variables()
        grads = tape.gradient(loss, variables)

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
            opt_topo.apply_gradients(zip(grads, variables))
        except ValueError:
            pass

        return sequence, loss


def loss_function(y_true, y_pred):
    probs = tf.math.softmax(y_pred, axis=-1)
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, probs))


epoch = 0
while True:

    for batch in train_dataset:

        sequence, loss = apply_batch(batch, loss_function, bweights=True, btopology=False)

        train_loss(loss)

    for batch in topology_dataset:

        sequence, loss = apply_batch(batch, loss_function, bweights=True, btopology=True)

        topology_loss(loss)

        num_active_neurons = 0
        num_active_connections = 0
        for weight_matrix, connection_mask_matrix, activation_mask, layer_norm, input_nodes_alive, output_nodes_alive in sequence:
            num_active_neurons += tf.where(input_nodes_alive).shape[0]
            num_active_connections += tf.where(tf.stack(connection_mask_matrix.values)).shape[0]

        active_neurons(num_active_neurons)
        active_connections(num_active_connections)
        mean_depth(len(sequence))

    for batch in test_dataset:

        sequence, loss = apply_batch(batch, loss_function, bweights=False, btopology=False)

        test_loss(loss)

    test.decay(0.001)

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
    with train_summary_writer.as_default():
        tf.summary.scalar('mean_depth', mean_depth.result(), step=epoch)
    active_neurons.reset_states()
    active_connections.reset_states()
    mean_depth.reset_states()

    epoch += 1

    top_vars = test.get_topologie_variables_grouped()
    for k in range(int(len(top_vars))):
        if k < len(params_writers):
            group = top_vars[k]
            if len(group) == 0:
                continue
            conn_params = tf.stack(group[0])
            writer = params_writers[k]
            writer(tf.reduce_mean(conn_params))

    for i in range(len(params_writers)):
        writer = params_writers[i]
        with train_summary_writer.as_default():
            tf.summary.scalar('connection_param_l{}'.format(i), writer.result(), step=epoch)
        writer.reset_states()

    if epoch % 5000 == 0:
        # test.save(checkpoint_path)
        # test = Network.load(checkpoint_path)
        pass
