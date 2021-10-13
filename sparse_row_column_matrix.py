import tensorflow as tf


class SparseRowColumnMatrix:
    def __init__(self, shape):
        assert len(shape) >= 2
        self.shape = shape
        self.rows = shape[0]
        self.columns = shape[1]

        self.b_rows = tf.constant(False, shape(self.rows, ), dtype=tf.bool)
        self.b_columns = tf.constant(False, shape(self.columns, ), dtype=tf.bool)
        self.rows = [None] * self.rows
        self.columns = [None] * self.columns

    def stack(self, b_rows, b_columns):
        pass

    def assign(self, value: tf.Tensor or tf.Variable, row: int, column: int):
        self.b_rows[row] = True
        self.b_columns[column] = True

        self.rows[row] = value
        self.columns[column] = value

    def __matmul__(self, b):
        a = self
        a_b_columns_altered = tf.logical_and(a.b_columns, b.b_rows)
        b_b_rows_altered = tf.logical_and(b.b_rows, a.b_columns)
        A = a.stack(a.b_rows, a_b_columns_altered)
        B = a.stack(b_b_rows_altered, b.b_columns)
        C = tf.matmul(A, B)


a = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=tf.Variable(tf.constant([1.0, 2.0], dtype=tf.float16)), dense_shape=[3, 4])
print(a)

