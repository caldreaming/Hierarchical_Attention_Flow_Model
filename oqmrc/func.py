import tensorflow as tf
import numpy as np

INF = 1e30


# 双向gru
class cudnn_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="cudnn_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            # 前向gru
            # 参数：层数，隐含单元数
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            # 后向gru
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            # init_state，shape=(1,batch_size,num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            # dropout层
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        # 转换输入的维度，0维与1维交换;并且增加一个维度（最外层[]作用）
        # 如输入shape为[2,3,4],outputs.shape=[[3,2,4]]
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = gru_fw(
                        outputs[-1] * mask_fw, initial_state=(init_fw, ))
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_axis=0, batch_axis=1)
                    out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw, ))
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_axis=0, batch_axis=1)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            # 连接每一层输出作为最终输出
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class native_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.rnn.GRUCell(num_units)
            gru_bw = tf.contrib.rnn.GRUCell(num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_axis=1, batch_axis=0)
                    out_bw, _ = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_axis=1, batch_axis=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        return res


def mru(input, seq_len, mask, mru_range, hidden):
    """
    :param input: [b,c,dim]
    :param seq_len: 1000
    :param mask: [b,c]
    :param mru_range: [1, 2, 4, 10, 25]
    :param hidden: 250
    """
    w_t = []
    N = input.get_shape().as_list()[0]
    dim = input.get_shape().as_list()[-1]
    for i, scale in enumerate(mru_range):
        in_i = input
        s_len = seq_len // scale
        if seq_len % scale is not 0:
            pad_size = scale - (seq_len % scale)
            padding = np.array([[0, 0], [0, pad_size], [0, 0]])
            in_i = tf.pad(input, padding)
            s_len = s_len + 1
        # shape=[N, s_len, scale*450]
        input_pad = tf.reshape(in_i, [N, -1, scale * dim])
        # contracted.shape=[N, s_len, hidden]
        contracted = tf.layers.dense(input_pad, units=hidden, activation=tf.nn.relu)
        # [N, s_len, 1, hidden] --> [N, s_len, scale, hidden] --> [N, w_size, hidden]
        expanded = tf.slice(tf.reshape(tf.tile(tf.expand_dims(contracted, 2),
                            [1, 1, scale, 1]), [N, s_len * scale, hidden]), [0, 0, 0], [N, seq_len, hidden])
        expanded = tf.tile(tf.expand_dims(tf.cast(mask, tf.float32), 2),
                           [1, 1, expanded.get_shape().as_list()[-1]]) * expanded
        w_t.append(expanded)

    w_t = tf.reshape(w_t, [N, seq_len, len(mru_range) * hidden])

    gate = tf.layers.dense(tf.layers.dense(w_t, units=250, activation=tf.nn.relu),
                           units=dim, activation=tf.nn.relu)
    gate = tf.nn.sigmoid(gate)
    z_t = tf.layers.dense(w_t, units=dim, activation=tf.nn.tanh)
    y_t = gate * z_t + (1 - gate) * input
    return y_t


class ptr_net:
    def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="ptr_net"):
        self.gru = tf.contrib.rnn.GRUCell(hidden)
        self.batch = batch
        self.scope = scope
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.dropout_mask = dropout(tf.ones(
            [batch, hidden], dtype=tf.float32), keep_prob=keep_prob, is_train=is_train)

    def __call__(self, init, match, d, mask):
        with tf.variable_scope(self.scope):
            d_match = dropout(match, keep_prob=self.keep_prob,
                              is_train=self.is_train)
            inp, logits1 = pointer(d_match, init * self.dropout_mask, d, mask)
            d_inp = dropout(inp, keep_prob=self.keep_prob,
                            is_train=self.is_train)
            _, state = self.gru(d_inp, init)
            tf.get_variable_scope().reuse_variables()
            _, logits2 = pointer(d_match, state * self.dropout_mask, d, mask)
            return logits1, logits2


def dropout(args, keep_prob, is_train, mode="recurrent"):
    """
    :param args: dropout matrix shape=[batch_size, 1, input_size_]
    :param keep_prob:
    :param is_train:
    :param mode:
    :return:
    """
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        # tf.cond():控制流程，如果第一个参数为True，执行第一个函数，否则执行第二个函数
        # 如果是train模式，使用dropout，否则mask矩阵全为1，即不使用dropout
        # noise_shape第二维为1，表示按照输入的第二维一致性dropout
        # 如：对于输入为(batch,word,vector_dim)的tensor,若一个单词被drop，整个句子即drop
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def pointer(inputs, state, hidden, mask, scope="pointer"):
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
            1, tf.shape(inputs)[1], 1]), inputs], axis=2)
        s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * inputs, axis=1)
        return res, s1


def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ"):
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * memory, axis=1)
        return res


def dot_attention(inputs, memory, mask, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]

        with tf.variable_scope("attention"):
            outputs = tf.matmul(d_inputs, tf.transpose(
                d_memory, [0, 2, 1])) / (inputs.get_shape().as_list()[-1] ** 0.5)
            # mask.shape:[batch,q_size]-->[batch,c_size,q_size]
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
            logits = tf.nn.softmax(softmax_mask(outputs, mask))
            # outputs.shape=[batch,c_size,word_dim]
            outputs = tf.matmul(logits, memory)
            # res.shape=[batch,c_size,2*word_dim]
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res * gate


def dense(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res


def bilinear(x, y, scope="bilinear"):
    """
    :param X: [batch, m, k]
    :param Y: [batch, n,l]
    :param scope: []
    :return res: [b,m,n]
    W = [batch,k,l]
    """
    with tf.variable_scope(scope):
        batch = x.get_shape().as_list()[0]
        dim_x = x.get_shape().as_list()[-1]
        dim_y = y.get_shape().as_list()[-1]
        W = tf.get_variable("W", [batch, dim_x, dim_y],
                            initializer=tf.random_uniform_initializer(
                                minval=0, maxval=None, seed=None, dtype=tf.float32))
        res = tf.matmul(x, W)
        # 0维Bacth不转， 1维和2维转
        res = tf.matmul(res, tf.transpose(y, perm=[0, 2, 1]))
        return res

