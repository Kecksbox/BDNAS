import tensorflow as tf


class SparseRowMatrix:
    def __init__(self, dense_shape, dtype=tf.float32):
        self.dense_shape = dense_shape
        self.dtype = dtype

        self.values = []
        self.indices = [False] * self.dense_shape[0]

    def assign(self, value, index):
        assert value.dtype == self.dtype

        row_already_filled = self.indices[index]
        previous_not_null_rows = tf.reduce_sum(tf.cast(self.indices[:index], dtype=tf.int32))
        self.indices[index] = True
        if not row_already_filled:
            self.values = self.values[:previous_not_null_rows] + [value] + self.values[previous_not_null_rows:]
        else:
            self.values = self.values[:previous_not_null_rows] + [value] + self.values[previous_not_null_rows + 1:]

    def mul_dense(self, b_dense):
        b_dense = tf.boolean_mask(b_dense, self.indices)
        c = tf.stack(self.values) * b_dense
        i_mask = tf.reduce_any(tf.cast(c, dtype=tf.bool), axis=list(range(1, len(c.shape))))
        c_masked = tf.reshape(tf.boolean_mask(c, i_mask), shape=[-1] + list(c.shape[1:]))
        si_where = tf.where(self.indices)
        new_indices = [False] * len(self.indices)
        for i, j in zip(si_where, i_mask):
            new_indices[tf.squeeze(i)] = tf.squeeze(j)
        result = SparseRowMatrix(dense_shape=[max(a, b) for a, b in zip(self.dense_shape, b_dense.shape)])
        result.values = tf.unstack(c_masked)
        result.indices = new_indices
        return result

    def matmul_dense(self, b_dense):
        r_values = tf.unstack(tf.matmul(tf.stack(self.values), b_dense))
        r = SparseRowMatrix(dense_shape=[self.dense_shape[0], b_dense.shape[1]])
        r.values = r_values
        r.indices = list.copy(self.indices)

        return r

    def operation(self, op):
        c = op(tf.stack(self.values))
        result = SparseRowMatrix(self.dense_shape)
        result.values = tf.unstack(c)
        result.indices = list.copy(self.indices)
        return result

    def to_dense(self, off_value, dtype: tf.DType):
        values = tf.cast(self.values, dtype)
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
            end = partitions[i+1]

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

    def cast(self, dtype):
        self.values = tf.unstack(tf.cast(tf.stack(self.values), dtype=dtype))
        self.dtype = dtype
        return self
