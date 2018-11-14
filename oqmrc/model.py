import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net, bilinear, mru


class Model(object):
    def __init__(self, config, batch, word_mat=None, trainable=True, opt=True):
        self.config = config
        # global_step传入train_op，会自动计数
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.c, self.q, self.alternatives, self.y, self.qa_id = batch.get_next()
        # is_train：设置这个参数是控制预测阶段不使用dropout层
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)
        # passage,query,候选答案长度
        # c.shape=(batch_size,para_limit)
        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.alter_mask = tf.cast(self.alternatives, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
        self.alter_len = tf.reduce_sum(tf.cast(self.alter_mask, tf.int32), axis=1)

        if opt:
            N = config.batch_size
            self.c_maxlen = tf.reduce_max(self.c_len)
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
            self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
            self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
            self.y = tf.slice(self.y, [0, 0], [N, 3])

        else:
            self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

        self.ready()

        if trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            # compute_gradients():It returns a list of (gradient, variable) pairs
            # where "gradient" is the gradient for "variable"
            # compute_gradients计算梯度；apply_gradients更新梯度
            # 相当于minimize(loss,global_step)函数的两个步骤
            grads = self.opt.compute_gradients(self.loss)
            # 解压分别保存
            gradients, variables = zip(*grads)
            # 梯度修剪
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            # 使用修剪后的梯度进行更新
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        config = self.config
        N, PL, QL, d= config.batch_size, self.c_maxlen, self.q_maxlen, config.hidden,
        gru = cudnn_gru if config.use_cudnn else native_gru

        # 词向量层
        with tf.variable_scope("emb"):
            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)
                alter_emb = tf.nn.embedding_lookup(self.word_mat, self.alternatives)

        # 编码层
        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            # [batch, c, 2*d*3]
            # 2：双向gru；3：连接3层的output作为最后的输出
            c = rnn(c_emb, seq_len=self.c_len)
            q = rnn(q_emb, seq_len=self.q_len)

        # with tf.variable_scope("mru_encoder"):
        #     c_m = mru(c, self.c_maxlen, self.c_mask, mru_range, 250)

        with tf.variable_scope("q2c"):
            # q2c.shape=[b,c,c.shape[-1]+q.shape[-1]]=[b,c,12d]
            q2c = dot_attention(c, q, mask=self.q_mask, keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=q2c.get_shape(
                ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            # v_c.shape=[b,c,2d]
            v_c = rnn(q2c, seq_len=self.c_len)

        with tf.variable_scope("q2o"):
            # alter.shape=[b,3,6d]
            alter = tf.layers.dense(alter_emb, units=6*d, activation=tf.nn.relu)
            # q2o.shape=[b,3,12d]
            q2o = dot_attention(alter, q, mask=self.q_mask, keep_prob=config.keep_prob, is_train=self.is_train)

        with tf.variable_scope("o2c"):
            # v_o.shape=[b,3,2d]
            v_o = tf.layers.dense(q2o, units=2*d, activation=tf.nn.relu)
            # o2c.shape=[b,c,4d]
            o2c = dot_attention(v_c, v_o, mask=self.alter_mask, keep_prob=config.keep_prob, is_train=self.is_train)
            r_c = tf.reduce_mean(o2c, axis=1, keepdims=True)
        with tf.variable_scope("predict"):
            # logits.shape=[b,3]
            logits = tf.reshape(bilinear(r_c, v_o), [N, v_o.get_shape().as_list()[1]])
            self.yp = tf.argmax(tf.nn.softmax(logits), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=tf.stop_gradient(self.y)
            )
            self.loss = tf.reduce_mean(losses)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step