import copy
import datetime
import os
import pickle
import random
import time
from pathlib import Path
from typing import List

import tensorflow as tf

random.seed(0)
tf.random.set_seed(0)
tf.get_logger().setLevel('ERROR')

min_conn_param = -12.0
max_conn_param = 12.0
conn_to_virtual_init_param = -4.0
conn_to_input_init_param = 0.0


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


def binary_gate_sigmoid(params, alive_vector, alive_probs_vector, train):
    tmp = tf.reshape(params, shape=(params.shape[0], -1))
    tmp = tf.math.sigmoid(tmp)
    probs = tmp

    if train:
        sample = binominal_sample_operation(probs)
    else:
        sample = tf.cast(tf.nn.relu(tf.sign(probs - tf.constant(1 / 2, shape=probs.shape))), dtype=tf.float32)

    sample = tf.minimum(tf.maximum(sample, 0.0), 1.0)
    return sample, probs * tf.expand_dims(
        alive_probs_vector, axis=-1)


@tf.custom_gradient
def check(weight_matrix, activation):
    def grad(upstream):
        wm_grad_st = tf.matmul(upstream, tf.transpose(tf.ones(activation.shape)))
        wm_grad_a = tf.matmul(upstream, tf.transpose(activation))
        wm_grad = tf.stack([wm_grad_st, wm_grad_a])
        a_grad = tf.matmul(tf.transpose(weight_matrix), upstream)
        return wm_grad, a_grad

    r = tf.matmul(weight_matrix, activation)

    return r, grad


@tf.custom_gradient
def check2(weight_matrix, mask):
    def grad(upstream):
        wm_grad_st, wm_grad_a = tf.unstack(upstream)
        w_grad = wm_grad_a * mask
        m_grad = weight_matrix * wm_grad_st
        return w_grad, m_grad

    r = weight_matrix * mask

    return r, grad


def reduce_connection_mask_matrix_operation(a, num_input_nodes):
    num_output_nodes = a.shape[1] - num_input_nodes

    a = tf.cast(a, dtype=tf.float32)
    mask = tf.reduce_sum(a, axis=0)
    mask_div = mask + tf.maximum((1 - 1 * mask), 0.0)
    result = (mask / mask_div)[:num_output_nodes]
    return result


class InputLayer:
    def __init__(self, dim_in: int):
        self.input = tf.Variable(0)
        self.dim_in = dim_in

    def assign(self, input):
        self.input.assign(input)


class Layer:
    def __init__(self, dim_in: int, dim_out: int, branching_factor: float = 2.0):
        self.dim_out = dim_out
        next_layer_output_dim = int(dim_out * branching_factor)
        self.num_valid_input_nodes = dim_in + int(dim_out * branching_factor)

        self.connection_parameter = tf.Variable(tf.concat([
            tf.constant(conn_to_virtual_init_param, shape=(dim_out, next_layer_output_dim), dtype=tf.float64),
            tf.constant(conn_to_input_init_param, shape=(dim_out, dim_in), dtype=tf.float64)
        ], axis=-1))

        self.weight_matrix = tf.Variable(tf.random.normal(shape=[dim_out, self.num_valid_input_nodes]))

        self.bias_vector = tf.Variable(tf.random.normal(shape=(dim_out, 1)))

        self.activation_parameter = tf.Variable(tf.constant(0.0, shape=(dim_out, 0)))

    def get_weight_variables(self):
        return [self.weight_matrix, self.bias_vector]

    def get_topologie_variables(self):
        return [self.connection_parameter, self.activation_parameter]

    def sample_topologie(self, output_nodes_alive: tf.Tensor, output_nodes_alive_prob, train):
        self.connection_parameter.assign(tf.clip_by_value(self.connection_parameter, min_conn_param, max_conn_param))

        connection_mask_matrix, connection_probs_matrix = binary_gate_sigmoid(
            tf.cast(self.connection_parameter, dtype=tf.float32),
            output_nodes_alive,
            output_nodes_alive_prob, train)

        weight_matrix = check2(self.weight_matrix, connection_mask_matrix)

        return weight_matrix, connection_mask_matrix, connection_probs_matrix


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

    def sample_topologie(self, max_depth, train: bool):
        sequence = []
        current_layer = self.output_layer
        output_nodes_alive = tf.constant(1.0, shape=[current_layer.dim_out], dtype=tf.float32)
        output_nodes_alive_probs = tf.constant(1.0, shape=[current_layer.dim_out], dtype=tf.float32)
        depth = 0
        while True:
            weight_matrix, connection_mask_matrix, connection_probs_matrix = current_layer.sample_topologie(
                output_nodes_alive, output_nodes_alive_probs, train)

            sequence.append(
                (weight_matrix, connection_mask_matrix, output_nodes_alive,
                 connection_probs_matrix,
                 output_nodes_alive_probs)
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

    def __call__(self, input, sequence, *args, **kwargs):
        num_input_nodes = self.input_layer.dim_in
        assert num_input_nodes == input.shape[1]

        first_hidden_layer_weights = sequence[-1][0]
        input = tf.transpose(input)
        activation = tf.constant(0.0, shape=(first_hidden_layer_weights.shape[1] - num_input_nodes, input.shape[1]))

        i = 1
        for weight_matrix, connection_mask_matrix, output_nodes_alive, _, _ in reversed(sequence):

            activation = tf.concat([activation, input], axis=0)

            activation = check(weight_matrix, activation)
            if i < len(sequence):
                activation = tf.keras.activations.relu(activation)
                i += 1
            activation = activation * tf.expand_dims(output_nodes_alive, axis=-1)

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
batch_size_sequence = 1
max_depth = 3

test = Network.load(checkpoint_path)
if test is None:
    test = Network(2, 1, 2.5)
opt_topo = tf.keras.optimizers.Adam(learning_rate=0.01)


#work_dir = os.environ["WORK"]
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + "dense_st_evaluation" + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
topology_loss = tf.keras.metrics.Mean('topology_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
active_neurons = tf.keras.metrics.Mean('active_neurons', dtype=tf.float32)
active_connections = tf.keras.metrics.Mean('active_connections', dtype=tf.float32)
connection_param_l1 = tf.keras.metrics.Mean('connection_param_l1', dtype=tf.float32)
connection_param_l2 = tf.keras.metrics.Mean('connection_param_l2', dtype=tf.float32)
connection_param_l3 = tf.keras.metrics.Mean('connection_param_l3', dtype=tf.float32)

params_writers = [connection_param_l1, connection_param_l2, connection_param_l3]

test_input = tf.constant([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])
test_output = tf.constant([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

topology_dataset = [
    (test_input, test_output)
]

epoch = 0
while True:
    for batch_x, batch_y in topology_dataset:
        with tf.GradientTape(persistent=False) as tape:
            loss = 0.0
            sequence = None
            for _ in range(batch_size_sequence):
                sequence = test.sample_topologie(max_depth=max_depth, train=True)

                result_train = test(batch_x, sequence)

                loss_train = tf.reduce_mean(tf.keras.losses.mean_squared_error(batch_y, result_train))

                log_loss = tf.cast(loss_train, dtype=tf.float64)
                loss += (1 / batch_size_sequence) * log_loss

            variables = test.get_topologie_variables() + test.get_weight_variables()
            grads = tape.gradient(loss, variables)

            try:
                opt_topo.apply_gradients(zip(grads, variables))
            except ValueError:
                pass

            topology_loss(loss_train)

            num_active_neurons = 0
            num_active_connections = 0
            for w, cmm, a, _, _ in sequence:
                num_active_neurons += tf.where(a).shape[0]
                num_active_connections += tf.where(cmm).shape[0]

            active_neurons(num_active_neurons)
            active_connections(num_active_connections)

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

    top_vars = test.get_topologie_variables()
    for k in range(int(len(top_vars) / 2)):
        if k < len(params_writers):
            writer = params_writers[k]
            writer(tf.reduce_mean(top_vars[int(k * 2)]))

    with train_summary_writer.as_default():
        tf.summary.scalar('connection_param_l1', connection_param_l1.result(), step=epoch)
    connection_param_l1.reset_states()
    with train_summary_writer.as_default():
        tf.summary.scalar('connection_param_l2', connection_param_l2.result(), step=epoch)
    connection_param_l2.reset_states()
    with train_summary_writer.as_default():
        tf.summary.scalar('connection_param_l3', connection_param_l3.result(), step=epoch)
    connection_param_l3.reset_states()

    if epoch % 5000 == 0:
        # test.save(checkpoint_path)
        # test = Network.load(checkpoint_path)
        pass
