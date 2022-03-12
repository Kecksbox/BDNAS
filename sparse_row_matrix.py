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
        value = tf.expand_dims(value, axis=0)
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
        result = SparseRowMatrix(dense_shape=[max(a, b) for a, b in zip(self.dense_shape, b_dense.shape)])
        result.value = c
        result.indices = self.indices
        return result

    def __add__(self, other):
        pass

    def __matmul__(self, other):
        assert self.dense_shape[1] == other.dense_shape[0]
        if self.value.shape[0] == 0 or other.value.shape[0] == 0:
            r_value = tf.zeros(shape=(self.value.shape[0], other.dense_shape[-1]))
            r_indices = self.indices
            r = SparseRowMatrix(dense_shape=[self.dense_shape[0], other.dense_shape[-1]])
            r.indices = r_indices
            r.value = r_value
            return r

        A = self.value
        B = other.value
        mask_A = other.indices
        masked_A = tf.boolean_mask(A, mask_A, axis=1)
        r_value = tf.matmul(masked_A, B)
        r_indices = self.indices
        r = SparseRowMatrix(dense_shape=[self.dense_shape[0], other.dense_shape[-1]])
        r.indices = r_indices
        r.value = r_value
        return r

    def operation(self, op):
        c = op(self.value)
        result = SparseRowMatrix(self.dense_shape)
        result.value = c
        result.indices = list.copy(self.indices)
        return result

    def concat(self, other):
        indices = self.indices + other.indices
        values = tf.concat([self.value, other.value], axis=0)
        res = SparseRowMatrix(dense_shape=[self.dense_shape[0] + other.dense_shape[0]] + self.dense_shape[1:])
        res.indices = indices
        res.value = values
        return res

    def concat_dense(self, other):
        other_sparse = SparseRowMatrix(dense_shape=other.shape, dtype=other.dtype)
        other_sparse.indices = [True] * other.shape[0]
        other_sparse.value = other
        return self.concat(
            other_sparse
        )

    def to_dense(self, off_value, dtype: tf.DType = tf.float32):
        values = tf.cast(self.value, dtype)
        # fill room between all active rows
        active_rows = tf.squeeze(tf.where(self.indices), axis=-1)
        # calculate blocks of active rows
        partitions = []
        partition = None
        for i in range(len(active_rows)):
            if partition is None:
                partition = [active_rows[i], active_rows[i]]
                partitions.append(partition)
            else:
                if active_rows[i] == partition[1] + 1:
                    partition[1] += 1
                else:
                    partition = [active_rows[i], active_rows[i]]
                    partitions.append(partition)

        partitions = [[-1, -1]] + partitions + [[self.dense_shape[0], self.dense_shape[0]]]

        partion_values = []
        row_counter = 0
        for i in range(len(partitions) - 1):
            start = partitions[i]
            end = partitions[i + 1]

            # retrieve active rows of this partition
            if start[-1] >= 0:
                num_elements = start[-1] - start[0] + 1
                new_row_counter = row_counter + num_elements
                start_partition_value = values[row_counter:new_row_counter]
                partion_values.append(start_partition_value)
                row_counter = new_row_counter

            distance = end[0] - start[-1] - 1
            partion_values.append(
                tf.constant(off_value, dtype=dtype, shape=(distance, self.dense_shape[-1]))
            )

        return tf.concat(partion_values, axis=0)
