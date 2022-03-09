import copy
import datetime
import json

import os

from PIL import Image, ImageDraw

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
import random
import time
from pathlib import Path
from typing import List, Callable

import tensorflow as tf
from keras.optimizer_v2.optimizer_v2 import _var_key

from sparse_row_matrix import SparseRowMatrix

tf.get_logger().setLevel('ERROR')

learning_rate = 0.001

conn_param_bias = 1 / 10000
min_conn_param = 0.0 + conn_param_bias
max_conn_param = 1.0 - conn_param_bias
categorical_smoothing_factor = 1 / 100
weight_bias = 0.001

inital_virtual_layer_output_node_alive_prob = 1 / 1000
inital_virtual_layer_output_node_alive_prob_to_input_layer = 1.0
decay_limit = min(inital_virtual_layer_output_node_alive_prob, 1 / 1000)


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
    tmp = tf.reshape(params.value, shape=(params.value.shape[0], -1))
    tmp2 = SparseRowMatrix(dense_shape=[params.dense_shape[0], tmp.shape[1]])
    tmp2.indices = params.indices
    tmp2.value = tmp
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


def categorical_smoothing_function(probs: tf.Tensor, factor: float) -> tf.Tensor:
    factor = tf.clip_by_value(factor, 0.0, 1.0)

    new_probs = probs * (1 - factor)
    new_probs += (factor / probs.shape[-1])
    new_probs /= tf.reduce_sum(new_probs, axis=-1, keepdims=True)

    return new_probs


def binary_gate_softmax(params):
    tmp = tf.math.softmax(params.value, axis=-1)
    tmp = categorical_smoothing_function(tmp, categorical_smoothing_factor)
    tmp = tf.reshape(tmp, shape=(params.value.shape[0], -1))
    tmp2 = SparseRowMatrix(dense_shape=[params.dense_shape[0], tmp.shape[1]])
    tmp2.indices = params.indices
    tmp2.value = tmp
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

    tmp = a.value
    mask = tf.reduce_sum(tmp, axis=0)
    mask_div = mask + tf.maximum((1 - 1 * mask), 0.0)
    tmp_reduced = (mask / mask_div)

    return tmp_reduced[:num_output_nodes], tmp_reduced


def assign_to_sparse_variable(var: SparseRowMatrix, value, index):
    row_already_filled = var.indices[index]
    previous_not_null_rows = tf.reduce_sum(tf.cast(var.indices[:index], dtype=tf.int32))
    value = tf.expand_dims(value, axis=0)

    optimizer = opt_topo
    weights = []
    slot_dict = optimizer._slots
    var_key = _var_key(var.value)
    if var_key in slot_dict:
        slot = optimizer._slots[var_key]
        weights = [slot['m'], slot['v']]

    var.indices[index] = True
    if not row_already_filled:
        new_value = tf.concat([var.value[:previous_not_null_rows], value, var.value[previous_not_null_rows:]],
                              axis=0)
        for w in weights:
            w._shape = [w.shape[0] + 1] + w.shape[1:]
            w.assign(
                tf.concat([w[:previous_not_null_rows], tf.zeros(shape=value.shape), w[previous_not_null_rows:]],
                          axis=0)
            )
    else:
        new_value = tf.concat(
            [var.value[:previous_not_null_rows], value, var.value[previous_not_null_rows + 1:]],
            axis=0)

        for w in weights:
            w.assign(
                tf.concat(
                    [w[:previous_not_null_rows], tf.zeros(shape=value.shape), w[previous_not_null_rows + 1:]],
                    axis=0)
            )

    var.value.assign(new_value)


def mul_by_alive_vector(target, alive_vector, init_function):
    rows_to_init = tf.logical_and(tf.logical_not(target.indices), alive_vector)
    for index in tf.where(rows_to_init):
        assign_to_sparse_variable(target, init_function(), tf.squeeze(index))

    # remove rows that are dead
    b_alive_masked = tf.boolean_mask(alive_vector, target.indices)
    target_masked = SparseRowMatrix(dense_shape=target.dense_shape)
    target_masked.value = tf.boolean_mask(tf.stack(target.value), b_alive_masked)
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
    tf.keras.layers.LeakyReLU(),
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
        self.connection_parameter.value = tf.Variable(tf.zeros(shape=[0, self.num_valid_input_nodes]),
                                                      shape=[None, self.num_valid_input_nodes])

        self.weight_matrix = SparseRowMatrix(dense_shape=[dim_out, self.num_valid_input_nodes])
        self.weight_matrix.value = tf.Variable(tf.zeros(shape=[0, self.num_valid_input_nodes]),
                                               shape=[None, self.num_valid_input_nodes])

        self.bias_matrix = SparseRowMatrix(dense_shape=[dim_out, 1])
        self.bias_matrix.value = tf.Variable(tf.zeros(shape=[0, 1]),
                                                      shape=[None, 1])

        self.activation_parameter = SparseRowMatrix(dense_shape=[dim_out, len(activation_function_catalog)])
        self.activation_parameter.value = tf.Variable(tf.zeros(shape=[0, len(activation_function_catalog)]),
                                                      shape=[None, len(activation_function_catalog)])

    def get_weight_variables(self):
        return [self.weight_matrix.value, self.bias_matrix.value]

    def get_topologie_variables(self):
        return [self.connection_parameter.value] + [
            self.activation_parameter.value]

    def get_topologie_sparse_variables(self):
        return [self.connection_parameter] + [self.activation_parameter]

    def init_connection_parameter(self):
        y = inital_virtual_layer_output_node_alive_prob / tf.cast(self.dim_out, dtype=tf.float32)
        return tf.Variable(
            tf.concat([
                tf.constant(y, shape=(self.dim_previous_layer,)),
                tf.constant(inital_virtual_layer_output_node_alive_prob_to_input_layer, shape=(self.dim_input_layer,))
            ], axis=0)
        )

    def init_weight_parameter(self):
        return tf.Variable(tf.random.uniform(shape=(self.num_valid_input_nodes,)) - 0.5)

    def init_bias_parameter(self):
        return tf.Variable(tf.random.uniform(shape=(1,)) - 0.5)

    def init_activation_paramter(self):
        return tf.Variable(tf.zeros(shape=(len(activation_function_catalog))))

    def sample_topologie(self, output_nodes_alive: SparseRowMatrix):
        local_min_conn_paramparam = max(
            min_conn_param,
            inital_virtual_layer_output_node_alive_prob / tf.cast(tf.cast(self.dim_out, dtype=tf.float32),
                                                                  dtype=tf.float32)
        )
        self.connection_parameter.value.assign(
            tf.clip_by_value(self.connection_parameter.value, local_min_conn_paramparam, max_conn_param)
        )

        self.connection_parameter, connection_parameter_masked = mul_by_alive_vector(self.connection_parameter,
                                                                                     output_nodes_alive,
                                                                                     init_function=self.init_connection_parameter)

        connection_mask_matrix, _ = binary_gate_sigmoid(connection_parameter_masked)

        self.weight_matrix, weight_matrix_masked = mul_by_alive_vector(self.weight_matrix,
                                                                       output_nodes_alive,
                                                                       init_function=self.init_weight_parameter)

        weight_matrix = SparseRowMatrix(dense_shape=weight_matrix_masked.dense_shape)
        weight_matrix.value = weight_matrix_masked.value * connection_mask_matrix.value
        weight_matrix_sign = tf.math.sign(weight_matrix.value)
        weight_matrix.value = weight_matrix.value + weight_matrix_sign * weight_bias
        weight_matrix.indices = weight_matrix_masked.indices

        self.activation_parameter, activation_parameter_masked = mul_by_alive_vector(self.activation_parameter,
                                                                                     output_nodes_alive,
                                                                                     init_function=self.init_activation_paramter)
        activation_mask, _ = binary_gate_softmax(activation_parameter_masked)

        self.bias_matrix, bias_matrix_masked = mul_by_alive_vector(self.bias_matrix,
                                             output_nodes_alive,
                                             init_function=self.init_bias_parameter)

        return weight_matrix, connection_mask_matrix, activation_mask, bias_matrix_masked


def apply_activation(activation, activation_mask):
    activation_map: List[SparseRowMatrix or None] = [None] * len(activation_function_catalog)
    for id in range(len(activation_function_catalog)):
        activation_map[id] = activation.operation(activation_function_catalog[id])

    a = tf.concat([tf.expand_dims(a.value, axis=-1) for a in activation_map], axis=-1)
    b = tf.expand_dims(activation_mask.value, axis=1)
    r = tf.reduce_sum(a * b, axis=-1)
    res = SparseRowMatrix(activation.dense_shape)
    res.value = r
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

    def get_topologie_sparse_variables(self):
        variables = []
        variables += self.output_layer.get_topologie_sparse_variables()
        for l in self.hidden_layers:
            variables += l.get_topologie_sparse_variables()
        return variables

    def get_topologie_variables_grouped(self):
        variables = []
        variables += [self.output_layer.connection_parameter.value]
        for l in self.hidden_layers:
            variables += [l.connection_parameter.value]
        return variables

    def get_hidden_layer(self, depth: int):
        if depth < 0:
            return self.output_layer
        assert depth < len(self.hidden_layers)
        return self.hidden_layers[depth]

    def sample_topologie(self):
        sequence = []
        current_layer = self.output_layer
        output_nodes_alive = tf.constant(1.0, shape=[current_layer.dim_out], dtype=tf.float32)
        depth = 0
        while True:
            weight_matrix, connection_mask_matrix, activation_mask, layer_norm = current_layer.sample_topologie(
                tf.cast(output_nodes_alive, dtype=tf.bool)
            )

            input_nodes_alive_layer, input_nodes_alive_full = reduce_connection_mask_matrix_operation(
                connection_mask_matrix, self.input_layer.dim_in)

            sequence.append(
                (
                    weight_matrix, connection_mask_matrix, activation_mask, layer_norm, input_nodes_alive_full,
                    output_nodes_alive
                )
            )

            if depth < len(self.hidden_layers) and tf.reduce_sum(input_nodes_alive_layer) > 0:
                current_layer = self.get_hidden_layer(depth)
                output_nodes_alive = input_nodes_alive_layer

                depth += 1
            else:
                break
        return sequence

    def calculate_alive_probabilities(self):
        results = []

        current_layer = self.output_layer
        output_nodes_alive_prob = tf.ones(shape=(current_layer.dim_out,))
        for virtual_layer in [self.output_layer] + self.hidden_layers:
            assert isinstance(virtual_layer, Layer)

            alive_probabilities_connections = virtual_layer.connection_parameter
            alive_probabilities_connections_masked = alive_probabilities_connections.mul_dense(
                tf.expand_dims(output_nodes_alive_prob, axis=-1)
            )

            results.append(
                (
                    output_nodes_alive_prob,
                    alive_probabilities_connections_masked,
                )
            )

            # calculate probability for next layer
            if tf.reduce_any(alive_probabilities_connections.indices):
                a = 1 - alive_probabilities_connections_masked.value
                num_input_nodes = self.input_layer.dim_in
                num_output_nodes = a.shape[1] - num_input_nodes

                output_nodes_alive_prob = 1 - tf.reduce_prod(a[:, :num_output_nodes], axis=0)
            else:
                output_nodes_alive_prob = tf.zeros(
                    shape=(virtual_layer.num_valid_input_nodes - self.input_layer.dim_in,)
                )

        return results

    def decay(self, limit: float):
        layers = [self.output_layer] + self.hidden_layers
        alive_probabilities = self.calculate_alive_probabilities()
        assert len(layers) == len(alive_probabilities)
        for i in range(len(layers)):
            virtual_layer = layers[i]
            assert isinstance(virtual_layer, Layer)
            output_nodes_alive_prob = alive_probabilities[i][0]

            # remove rows by limit
            tf.print(tf.reduce_mean(output_nodes_alive_prob))
            decay_save_mask = tf.greater_equal(output_nodes_alive_prob, limit)

            # remove connection paramter
            target = virtual_layer.connection_parameter
            b_alive_mask = tf.boolean_mask(decay_save_mask, target.indices)
            target.value.assign(
                tf.boolean_mask(target.value, b_alive_mask)
            )
            target.indices = tf.unstack(tf.logical_and(decay_save_mask, target.indices))
            # remove estimates from adam optimizer
            optimizer = opt_topo
            slot_dict = optimizer._slots
            var_key = _var_key(target.value)
            if var_key in slot_dict:
                slot = optimizer._slots[var_key]
                for key in ['m', 'v']:
                    slot[key] = tf.Variable(
                        tf.boolean_mask(slot[key], b_alive_mask)
                    )

            # remove node parameter (activation)
            target = virtual_layer.activation_parameter
            b_alive_mask = tf.boolean_mask(decay_save_mask, target.indices)
            target.value.assign(
                tf.boolean_mask(target.value, b_alive_mask)
            )
            target.indices = tf.unstack(tf.logical_and(decay_save_mask, target.indices))
            # remove estimates from adam optimizer
            optimizer = opt_topo
            slot_dict = optimizer._slots
            var_key = _var_key(target.value)
            if var_key in slot_dict:
                slot = optimizer._slots[var_key]
                for key in ['m', 'v']:
                    slot[key] = tf.Variable(
                        tf.boolean_mask(slot[key], b_alive_mask)
                    )

    def __call__(self, input_batch, sequence, *args, **kwargs):
        sequence_length = len(sequence)
        input_batch, batch_activation_history = input_batch

        num_input_nodes = self.input_layer.dim_in
        assert num_input_nodes == input_batch.shape[1]

        first_hidden_layer_weights = sequence[-1][0]
        input = tf.transpose(input_batch)
        activation = SparseRowMatrix(
            dense_shape=[first_hidden_layer_weights.dense_shape[1] - num_input_nodes, input.shape[1]])

        i = 1
        for weight_matrix, connection_mask_matrix, activation_mask, bias, input_nodes_alive, _ in reversed(
                sequence):
            activation = activation.concat_dense(input)

            activation = activation.mul_dense(tf.expand_dims(input_nodes_alive, axis=-1))

            start = time.time()
            activation = weight_matrix.__matmul__(activation)
            end = time.time()
            # tf.print("matmul took: {}".format(end - start))

            activation.value += bias.value

            if i < len(sequence):
                activation = apply_activation(activation, activation_mask)

            i += 1

        activation = activation.to_dense(0.0, tf.float32)

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
dim_input = 2
dim_output = 1
if test is None:
    test = Network(dim_input, dim_output, [
        Layer(dim_input, 100, 100),
        Layer(dim_input, 100, 100),
        Layer(dim_input, 100, 100),
        Layer(dim_input, 0, 100),
    ])

opt_topo = tf.keras.optimizers.Adam(learning_rate=learning_rate)

decay_interval = 100
log_interval = 100
top_vis_intervall = 500

# work_dir = os.environ["WORK"]
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + "sparse_node_mask_classic_3" + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
topology_loss = tf.keras.metrics.Mean('topology_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
active_neurons = tf.keras.metrics.Mean('active_neurons', dtype=tf.float32)
active_connections = tf.keras.metrics.Mean('active_connections', dtype=tf.float32)
mean_depth = tf.keras.metrics.Mean('mean_depth', dtype=tf.float32)
memory_used = tf.keras.metrics.Mean('memory_used', dtype=tf.float32)

tp_cp_path = f'logs/topology_checkpoints/{time.strftime("%Y%m%d-%H%M%S")}/'
if os.path.exists(tp_cp_path):
    os.remove(tp_cp_path)
if not os.path.exists(tp_cp_path):
    os.makedirs(tp_cp_path)

layers = [test.output_layer] + test.hidden_layers
params_writers = []
for i in range(len(layers)):
    params_writers.append(
        tf.keras.metrics.Mean('connection_param_l{}'.format(i), dtype=tf.float32)
    )
uncertainty_writers = []
for i in range(len(layers)):
    uncertainty_writers.append(
        tf.keras.metrics.Mean('connection_uncertainty_l{}'.format(i), dtype=tf.float32)
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

batches_topology = 5
batch_size_topology = 100

batches_test = 0
batch_size_test = 200

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


def clip_grads_by_global_norm(grads, norm: float):
    types = [None] * len(grads)
    conv_grads = [None] * len(grads)
    for grad_index in range(len(grads)):
        grad = grads[grad_index]
        if grad is None:
            continue
        types[grad_index] = grad.dtype
        conv_grads[grad_index] = tf.cast(grad, dtype=tf.float64)
    grads, _ = tf.clip_by_global_norm(conv_grads, norm)
    for grad_index in range(len(grads)):
        type = types[grad_index]
        if type is None:
            continue
        grads[grad_index] = tf.cast(grads[grad_index], dtype=type)

    return grads


def apply_gradients(grads, variables):
    try:
        slot_dict = opt_topo._slots
        for var in variables:
            var._shape = tf.shape(var)
            var_key = _var_key(var)
            if var_key in slot_dict:
                var_slot = slot_dict[var_key]
                var_m = var_slot['m']
                if var_m.shape[0] != var.shape[0]:
                    # remove m and v
                    # This case should never occure in practice.
                    raise AssertionError("Unexpexted shape difference in optimizer estimates.")

        opt_topo.apply_gradients(zip(grads, variables))
        for var in variables:
            var._shape = tf.TensorShape([None]).__concat__(var._shape[1:])
    except ValueError:
        pass


def apply_batch(batch, loss_function: Callable, train_weight: bool, train_topology: bool):
    batch_x, batch_y = batch

    train = train_weight or train_topology
    with tf.GradientTape() as topology_tape:
        if not train:
            topology_tape.stop_recording()
        ## -----------------------------topology part--------------------------------------------------------
        sequence = test.sample_topologie()

        result_train = test(batch_x, sequence)

        loss = loss_function(batch_y, result_train)

        variables = []
        if train_weight:
            variables += test.get_weight_variables()
        if train_topology:
            variables += test.get_topologie_variables()
        grads = topology_tape.gradient(loss, variables)

        apply_gradients(grads, variables)

        return sequence, loss


def loss_function(y_true, y_pred):
    probs = tf.math.softmax(y_pred, axis=-1)
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, probs))

#def loss_function(y_true, y_pred):
#    return tf.keras.losses.mean_squared_error(y_true, y_pred)


epoch = 0
while True:

    for batch in train_dataset:
        sequence, loss = apply_batch(batch, loss_function, train_weight=True, train_topology=False)

        train_loss(loss)

    for batch in topology_dataset:

        start = time.time()
        sequence, loss = apply_batch(batch, loss_function, train_weight=True, train_topology=True)
        end = time.time()
        # tf.print("apply_batch took: {}".format(end - start))

        topology_loss(loss)

        num_active_neurons = 0
        num_active_connections = 0
        for weight_matrix, connection_mask_matrix, activation_mask, layer_norm, input_nodes_alive, output_nodes_alive in sequence:
            num_active_neurons += tf.where(input_nodes_alive).shape[0]
            num_active_connections += tf.where(connection_mask_matrix.value).shape[0]

        active_neurons(num_active_neurons)
        active_connections(num_active_connections)
        mean_depth(len(sequence))

    for batch in test_dataset:
        sequence, loss = apply_batch(batch, loss_function, train_weight=False, train_topology=False)

        test_loss(loss)

    start = time.time()
    if epoch % decay_interval == 0:
        test.decay(decay_limit)
    end = time.time()
    # tf.print("decay took: {}".format(end - start))

    top_vars = test.get_topologie_variables_grouped()
    for k in range(int(len(top_vars))):
        if k < len(params_writers):
            group = top_vars[k]
            conn_params = group
            writer = params_writers[k]
            mean = tf.reduce_mean(conn_params)
            if tf.math.is_nan(mean):
                continue
            writer(tf.reduce_mean(conn_params))

            writer = uncertainty_writers[k]
            uncertainty = tf.reduce_sum(tf.minimum(tf.math.square(1 - conn_params), tf.math.square(conn_params)))
            if tf.math.is_nan(uncertainty):
                continue
            writer(tf.reduce_mean(uncertainty))

    # memory consumtion
    all_vars = test.get_topologie_variables() + test.get_weight_variables()
    memory_size = 0
    for var in all_vars:
        memory_size += tf.size(var)
    memory_used(memory_size)

    if epoch % log_interval == 0:

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

        with train_summary_writer.as_default():
            tf.summary.scalar('memory_used', memory_used.result(), step=epoch)
        memory_used.reset_states()

        for i in range(len(params_writers)):
            writer = params_writers[i]
            with train_summary_writer.as_default():
                tf.summary.scalar('connection_param_l{}'.format(i), writer.result(), step=epoch)
            writer.reset_states()
        for i in range(len(uncertainty_writers)):
            writer = uncertainty_writers[i]
            with train_summary_writer.as_default():
                tf.summary.scalar('connection_uncertainty_l{}'.format(i), writer.result(), step=epoch)
            writer.reset_states()

    if epoch % top_vis_intervall == 0:
        tp_vars = test.get_topologie_sparse_variables()
        layer_vars = []
        max_nodes = dim_input
        for i in range(int(len(tp_vars) / 2)):
            layer_vars.append((
                tp_vars[i * 2].value.numpy(),
                tp_vars[i * 2 + 1].value.numpy(),
                tp_vars[i * 2 + 1].indices,
            ))
            activation_vars = layer_vars[-1][1]
            if activation_vars.shape[0] > max_nodes:
                max_nodes = activation_vars.shape[0]

        alive_probabilities = test.calculate_alive_probabilities()

        # create graph
        image_size = (1000, 1000)
        image = Image.new('RGBA', image_size)
        draw_image = ImageDraw.Draw(image, 'RGBA')
        draw_image.rectangle((0, 0, image_size[0], image_size[1]), fill='black')

        num_layers = len(layer_vars) + 1
        layer_box_max_width = image_size[0] / num_layers
        layer_box_max_hight = image_size[1] / max_nodes
        node_size = min(layer_box_max_width, layer_box_max_hight) / 3
        width_padding = (layer_box_max_width - node_size) / 2
        height_padding = (layer_box_max_hight - node_size) / 2

        w_space_needed = node_size * num_layers + (num_layers - 1) * width_padding
        outer_width_padding = (image_size[0] - w_space_needed) / 2

        old_centers = []
        known_nodes_in_previous_layer = []
        for i in range(num_layers):
            new_centers = []
            if i == 0:
                num_nodes = dim_input
                num_connections = 0
                known_nodes_in_current_layer = [True] * dim_input
                node_alive_probability = [1.0] * dim_input
                conn_alive_probability = SparseRowMatrix([0])
            else:
                conn_vars, activation_vars, known_nodes_in_current_layer = layer_vars[-i]
                node_alive_probability, conn_alive_probability = alive_probabilities[-i]

                if i < num_layers - 1:
                    num_nodes = activation_vars.shape[0] + dim_input
                    num_connections = conn_vars.shape[-1]
                    known_nodes_in_current_layer = known_nodes_in_current_layer + [True] * dim_input
                    node_alive_probability = tf.concat([node_alive_probability, tf.ones(shape=(dim_input,))], axis=0)
                else:
                    num_nodes = activation_vars.shape[0]
                    num_connections = conn_vars.shape[-1]
                    known_nodes_in_current_layer = known_nodes_in_current_layer
                    node_alive_probability = node_alive_probability

            known_nodes_in_current_layer_ids = tf.where(known_nodes_in_current_layer)

            h_space_needed = num_nodes * node_size + (num_nodes - 1) * height_padding
            outer_height_padding = (image_size[1] - h_space_needed) / 2

            offset = [
                outer_width_padding + i * (width_padding + node_size),
                outer_height_padding,
            ]
            for j in range(num_nodes):
                node_id = tf.squeeze(known_nodes_in_current_layer_ids[j])
                alpha = int(node_alive_probability[node_id] * 255)

                transp_tmp = Image.new('RGBA', image_size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(transp_tmp, "RGBA")

                local_center = (offset[0] + (node_size / 2), offset[1] + (node_size / 2))
                new_centers.append(local_center)
                draw.ellipse((offset[0], offset[1], offset[0] + node_size, offset[1] + node_size),
                             fill=(255, 255, 255, alpha))

                image.paste(Image.alpha_composite(image, transp_tmp))

                offset[1] += node_size + height_padding

                if node_id >= conn_alive_probability.dense_shape[0]:
                    continue

                if conn_alive_probability.indices[node_id]:
                    value_index = tf.reduce_sum(tf.cast(conn_alive_probability.indices[:node_id], dtype=tf.int32))
                    node_conn_alive_probs = conn_alive_probability.value[value_index]
                else:
                    node_conn_alive_probs = tf.zeros(shape=(num_connections,))

                start_center = local_center
                center_index = 0
                for z in range(num_connections):
                    if not known_nodes_in_previous_layer[z]:
                        continue
                    target_center = old_centers[center_index]
                    center_index += 1

                    transp_tmp = Image.new('RGBA', image_size, (0, 0, 0, 0))
                    draw = ImageDraw.Draw(transp_tmp, "RGBA")

                    alpha2 = int(node_conn_alive_probs[z] * 255)

                    draw.line((start_center[0], start_center[1], target_center[0], target_center[1]),
                              fill=(255, 255, 255, alpha2))
                    image.paste(Image.alpha_composite(image, transp_tmp))

            old_centers = new_centers
            known_nodes_in_previous_layer = known_nodes_in_current_layer

        image.save(tp_cp_path + f'alive_{epoch}.png')

    if epoch % 5000 == 0:
        # test.save(checkpoint_path)
        # test = Network.load(checkpoint_path)
        pass

    epoch += 1
