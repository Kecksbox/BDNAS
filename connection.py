import datetime
import random
from typing import List

import tensorflow as tf

tf.get_logger().setLevel('ERROR')

tf.random.set_seed(0)


class InputNode:
    def __init__(self):
        self.input = tf.Variable(0.0, dtype=tf.float32)
        self.outgoing: List[Connection] = []
        self.dependencies = set()

    def update_dependency(self):
        return

    def reset(self):
        return

    def set(self, value):
        self.input.assign(value)

    def get_variables(self):
        return [self.input]

    def __call__(self, *args, **kwargs):
        return self.input


node_set = set()
node_counter = 0


class Node:
    def __init__(self):
        self.incomming: List[Connection] = []
        self.outgoing: List[Connection] = []
        self.activation_function = tf.keras.activations.relu

        self.value = None
        self.vars = None

        self.visited = 0

        self.dependencies = set()

        global node_set
        node_set.add(self)

        global node_counter
        self.node_id = node_counter
        node_counter += 1

    def reset(self):
        self.value = None
        self.vars = None

        for c in self.incomming:
            c.reset()

        if self.visited == 0:
            self.__del__()
            for n in node_set:
                n.update_dependency()

    def update_dependency(self):
        for c in self.incomming:
            self.dependencies.add(c.source)
            c.source.update_dependency()
            self.dependencies.update(c.source.dependencies)

    def get_variables(self):
        if self.vars is None:
            self.vars = []
            for i in self.incomming:
                self.vars += i.get_variables()
        return self.vars

    def __call__(self, *args, **kwargs):
        self.visited += 1
        if self.value is None:
            activation = 0.0
            for conn in self.incomming:
                activation += conn()

            self.value = activation
        return self.value

    def __del__(self):
        for c in self.incomming + self.outgoing:
            c.__del__()
        node_set.remove(self)
        self.incomming = None
        self.outgoing = None


@tf.custom_gradient
def binary_gate(params):
    probs = tf.math.softmax(params)
    global max_new_nodes
    global new_node_counter
    if max_new_nodes <= new_node_counter:
        probs *= tf.constant([1.0, 1.0, 0.0])
        probs /= tf.reduce_sum(probs)
    sample = tf.squeeze(tf.random.categorical(logits=tf.math.log(tf.expand_dims(probs, axis=0)), num_samples=1))
    sample = tf.one_hot(sample, depth=probs.shape[-1])

    def grad(upstream):
        return upstream * probs

    return sample, grad


def create_alternative(original_target: Node):
    global node_set
    global new_node_counter
    new_node_counter += 1
    new_node = Node()
    alternative_connection: Connection or None = None
    for n in node_set:
        if n == new_node:
            continue
        if original_target not in n.dependencies and n != original_target:
            Connection(n, new_node)
        else:
            c = Connection(new_node, n)
            c.topology_params.assign(tf.constant([1.0, 1.0, 1.0], shape=(3,)))
            if n == original_target:
                alternative_connection = c
    for n in node_set:
        n.update_dependency()
    return alternative_connection


max_new_nodes = 1
new_node_counter = 0


class Connection:
    def __init__(self, source: Node, target: Node):
        self.source = source
        self.target = target

        target.incomming.append(self)
        source.outgoing.append(self)

        self.weight = tf.Variable(tf.random.normal(shape=()), dtype=tf.float32)
        self.topology_params = tf.Variable(tf.constant([1.0, 1.0, 1.0], shape=(3,)))

        self.alternative: Connection or None = None
        self.parent: Connection or None = None

    def reset(self):
        self.source.reset()

    def get_variables(self):
        vars = [self.weight, self.topology_params] + self.source.get_variables()
        return vars

    def __call__(self, *args, **kwargs):
        sample = binary_gate(self.topology_params)
        options = [0.0] * sample.shape[-1]
        sampled_choice = tf.argmax(sample)
        if sampled_choice == 0:
            options[0] = 0.0
        elif sampled_choice == 1:
            options[1] = self.weight * self.source()
        elif sampled_choice == 2:
            if self.alternative is None:
                self.alternative = create_alternative(self.target)
                self.alternative.parent = self
            options[2] = self.alternative()
        return tf.reduce_sum(sample * options)

    def __del__(self):
        if self.source.outgoing is not None:
            self.source.outgoing.remove(self)
        if self.target.incomming is not None:
            self.target.incomming.remove(self)
        if self.parent is not None:
            self.parent.alternative = None

        self.weight = None
        self.alternative = None

class Network:
    def __init__(self, inputs: int, outputs: int):
        self.inputs = []
        for _ in range(inputs):
            self.inputs.append(InputNode())
        self.outputs = []
        for _ in range(outputs):
            self.outputs.append(Node())

        for inp in self.inputs:
            for o in self.outputs:
                Connection(inp, o)

        for n in node_set:
            n.update_dependency()

    def reset(self):
        for o in self.outputs:
            o.reset()

        for n in node_set:
            n.visited = 0

    def get_variables(self):
        vars = []
        for o in self.outputs:
            vars += o.get_variables()
        return vars

    def __call__(self, inputs, *args, **kwargs):
        global new_node_counter
        new_node_counter = 0

        for i in range(inputs.shape[0]):
            self.inputs[i].set(inputs[i])

        result = []
        for o in self.outputs:
            result.append(o(0))

        return tf.stack(result)


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

test = Network(784, 10)
input = tf.constant([1.0, 2.0, 3.0])
opt = tf.keras.optimizers.Adam(0.1)

mnist = tf.keras.datasets.mnist
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.cast(tf.reshape(x_train, shape=(-1, x_train.shape[-1] * x_train.shape[-2])), dtype=tf.float32)
x_test = tf.cast(tf.reshape(x_test, shape=(-1, x_test.shape[-1] * x_test.shape[-2])), dtype=tf.float32)

epoch = 0
while True:
    train_batch = random.choices(range(0, x_train.shape[0]), k=100)
    train_batch = [x_train[i] for i in train_batch]

    with tf.GradientTape() as tape:
        for e in train_batch:
            result = tf.math.softmax(test(e))
            loss = tf.keras.losses.mean_squared_error(y_train, result)
            vars = test.get_variables()
            print(len(vars))
            print(len(node_set))
            grads = tape.gradient(loss, vars)
            opt.apply_gradients(zip(grads, vars))
            print(result)

            train_loss(loss)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
        epoch += 1
        train_loss.reset_states()


        test.reset()
