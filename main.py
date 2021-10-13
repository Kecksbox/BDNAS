import tensorflow as tf

a = tf.Variable(2.0)
b = tf.Variable(4.0)

output = tf.multiply(a, b)

print(a)
print(output.numpy())

a.assign(1.0)

print(a)
print(output.numpy())