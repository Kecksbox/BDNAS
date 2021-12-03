from typing import List

import tensorflow as tf


class SparseRowMatrix:
    def __init__(self, dense_shape, dtype=tf.float32):
        self.dense_shape = dense_shape
        self.value = tf.constant(0.0, dtype=dtype, shape=[0] + self.dense_shape[1:])
        self.indices = [False] * self.dense_shape[0]

    def assign(self, value, index):
        row_already_filled = self.indices[index]
        previous_not_null_rows = tf.reduce_sum(tf.cast(self.indices[:index], dtype=tf.int32))
        self.indices[index] = True
        if not row_already_filled:
            new_value = tf.concat([self.value[:previous_not_null_rows], value, self.value[previous_not_null_rows:]],
                                  axis=0)
        else:
            new_value = tf.concat(
                [self.value[:previous_not_null_rows], value, self.value[previous_not_null_rows + 1:]],
                axis=0)
        if isinstance(self.value, tf.Variable):
            self.value.assign(new_value)
        else:
            self.value = new_value

    def mul_dense(self, b_dense):
        b_dense = tf.boolean_mask(b_dense, self.indices)
        c = self.value * b_dense
        i_mask = tf.reduce_any(tf.cast(c, dtype=tf.bool), axis=list(range(1, len(c.shape))))
        c_masked = tf.reshape(tf.boolean_mask(c, i_mask), shape=[-1] + list(c.shape[1:]))
        si_where = tf.where(self.indices)
        new_indices = [False] * len(self.indices)
        for i, j in zip(si_where, i_mask):
            new_indices[tf.squeeze(i)] = tf.squeeze(j)
        result = SparseRowMatrix(dense_shape=[max(a, b) for a, b in zip(self.dense_shape, b_dense.shape)])
        result.value = c_masked
        result.indices = new_indices
        return result

    def operation(self, op):
        c = op(self.value)
        result = SparseRowMatrix(self.dense_shape)
        result.value = c
        result.indices = list.copy(self.indices)
        return result

    def to_dense(self, off_value):
        old_state = list.copy(self.indices), self.value
        for index in tf.where(tf.logical_not(self.indices)):
            self.assign(
                tf.expand_dims(tf.constant(off_value, shape=self.dense_shape[1:]), axis=0),
                tf.squeeze(index)
            )
        result = self.value
        self.indices, self.value = old_state
        return result

    def to_tf_sparse(self):
        values = []
        indices = []
        for j, index in enumerate(tf.where(self.indices)):
            tmp_sparse_row = tf.sparse.from_dense(self.value[j])
            values.append(tmp_sparse_row._values)
            row_indices = tf.squeeze(tf.transpose(tf.stack([
                tf.constant(index, shape=(tmp_sparse_row.indices.shape[0], 1), dtype=tf.int64),
                tmp_sparse_row.indices,
            ], axis=0)), axis=0)
            indices.append(row_indices)
        result = tf.SparseTensor(indices=tf.constant(0, shape=[0, 2], dtype=tf.int64),
                                 values=tf.constant(0, shape=[0, ], dtype=tf.float32),
                                 dense_shape=self.dense_shape)
        result._values = tf.concat(values, axis=0)
        result._indices = tf.concat(indices, axis=0)
        return result

    @staticmethod
    def from_dense(a):
        return SparseRowMatrix(a.shape, tf.unstack(a), list(range(a.shape[0])))

    def __add__(self, other):
        new_sparse_matrix = SparseRowMatrix(shape=self.shape, rows=[], row_keys=[])
        new_sparse_matrix.rows = list.copy(self.rows)
        new_sparse_matrix.keys = list.copy(self.keys)
        for j in range(len(other.keys)):
            other_key = other.keys[j]
            other_row = other.rows[j]
            match_found = False
            for i in range(len(new_sparse_matrix.keys)):
                key = new_sparse_matrix.keys[i]
                if key == other_key:
                    new_sparse_matrix.rows[i] = new_sparse_matrix.rows[i] + other_row
                    match_found = True
                    break
            if not match_found:
                new_sparse_matrix.assign(other_row, other_key)
        return new_sparse_matrix

    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0]
        if self.is_empty() or other.is_empty():
            return SparseRowMatrix(shape=(self.shape[0], other.shape[-1]), rows=[], row_keys=[])

        A = tf.stack(self.rows)
        valid_columns = []
        for row_key in other.keys:
            valid_columns.append(row_key)
        A = tf.gather(A, indices=valid_columns, axis=1)
        B = tf.stack(other.rows)
        result = tf.matmul(A, B)
        return SparseRowMatrix(shape=(self.shape[0], other.shape[-1]), rows=tf.unstack(result), row_keys=self.keys)

    def __mul__(self, other):
        C_row_keys = []
        A_rows = []
        B_rows = []
        for i in range(len(self.keys)):
            key = self.keys[i]
            for j in range(len(other.keys)):
                other_key = other.keys[j]
                if other_key == key:
                    C_row_keys.append(key)
                    A_rows.append(self.rows[i])
                    B_rows.append(other.rows[j])
                    break
        result = tf.stack(A_rows) * tf.stack(B_rows)
        return SparseRowMatrix(shape=self.shape, rows=tf.unstack(result), row_keys=C_row_keys)


"""
test = SparseRowMatrix(dense_shape=(3, 2))
test.assign(tf.Variable(tf.constant(1.0, shape=(1, 2))), 1)
test.assign(tf.Variable(tf.constant(0.5, shape=(1, 2))), 2)
test.assign(tf.constant(-0.1, shape=(1, 2)), 0)
"""
