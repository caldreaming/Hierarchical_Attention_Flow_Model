import tensorflow as tf
import json
import numpy as np
import os
from tqdm import tqdm
from util import get_record_parser, get_batch_dataset, get_dataset, convert_tokens, evaluate
from model import Model
import logging

def train(config):

    with open(config.word_emb_file, "r", encoding="utf-8") as fh:
        # 载入词向量矩阵，shape=(存在词向量的词数+2,300)  //2是指NULL,OOV
        word_mat = np.array(json.load(fh), dtype=np.float32)

    with open(config.train_eval_file, "r", encoding="utf-8") as fh:
        # 载入评估数据
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r", encoding="utf-8") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        # dev集样本数
        meta = json.load(fh)

    dev_total = meta["total"]

    print("Building model ...")
    # 解析器，解析单个example的函数
    parser = get_record_parser(config)
    # 使用解析器批量获取record文件中的数据
    train_dataset = get_batch_dataset(config.train_record_file, parser, config)
    dev_dataset = get_batch_dataset(config.dev_record_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    # 通过tf.data.Iterator.from_string_handle来定义一个 feedable iterator，达到切换数据集的目的
    # 创建迭代器访问dataset，handle参数控制使用哪个迭代器（对应一个dataset）
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes
    )
    # 为trainset，devset各自创建迭代器
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    #加载模型
    model = Model(config, iterator, word_mat)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    loss_save = 100.0
    patience = 0.0
    lr = config.init_lr
    # 训练过程的loss保存到文件中
    logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                        filename=config.log_file,
                        format='%(message)s'  # 日志格式
                        )
    with tf.Session(config=sess_config) as sess:
        # log&summary
        writer = tf.summary.FileWriter(config.log_dir)
        # 保存model
        saver = tf.train.Saver(max_to_keep=config.max_to_keep)
        # sess.run()获取不同数据集的handel，feed in的方式切换数据集
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        # 载入之前训练的model
        ckpt = tf.train.get_checkpoint_state(config.save_dir)
        if not config.restart and ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
            metrics, _ = evaluate_batch(
                model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)

            loss_save = metrics["loss"]
            logging.info("Restore from saved model,saved loss is {},saved acc is {}".format(loss_save, metrics["acc"]))
        else:
            sess.run(tf.global_variables_initializer())
        # 给model传参
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
        for _ in tqdm(range(1, config.num_steps + 1)):
            global_step = sess.run(model.global_step) + 1
            # train一个batch
            loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                handle: train_handle})
            # 周期地保存loss
            if global_step % config.period == 0:
                # 记录loss
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
            # checkpoint表示每经过一定数量的迭代后就进行一次evaluation
            if global_step % config.checkpoint == 0:
                # 关闭训练模式
                sess.run(tf.assign(model.is_train,
                                   tf.constant(False, dtype=tf.bool)))

                logging.info("At {} step,model loss is {}".format
                             (global_step, loss))
                # evaluation
                metrics, summ = evaluate_batch(
                    model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                for s in summ:
                    writer.add_summary(s, global_step)
                logging.info("At {} step,train loss is {},train acc is {}".format
                             (global_step, metrics["loss"], metrics["acc"]))
                metrics, summ = evaluate_batch(
                    model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)

                # 开启训练模式，继续训练
                sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))

                dev_loss = metrics["loss"]
                logging.info("At {} step,dev loss is {},dev acc is {}".format
                             (global_step, dev_loss, metrics["acc"]))
                # print("At {} step,train loss is {},dev loss is {}".format(global_step, loss, dev_loss), file=loss_log)
                if dev_loss < loss_save:
                    logging.info("saving model at step {},dev loss is {},dev acc is {}...".format
                                 (global_step, dev_loss, metrics['acc']))
                    # print("saving model,dev loss is {}...".format(global_step, dev_loss))
                    loss_save = dev_loss
                    patience = 0
                    # 保存model
                    filename = os.path.join(
                        config.save_dir, "model_{}.ckpt".format(global_step))
                    saver.save(sess, filename)
                else:
                    patience += 1
                    if patience > config.early_stop:
                        break
                for s in summ:
                    writer.add_summary(s, global_step)
                # 将未满的缓冲区数据刷到文件中
                writer.flush()


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp = sess.run(
            [model.qa_id, model.loss, model.yp], feed_dict={handle: str_handle})
        # logging.info(str(yp))
        answer_dict_, _ = convert_tokens(
            eval_file, qa_id.tolist(), yp)
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    acc_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/acc".format(data_type), simple_value=metrics["acc"]), ])

    return metrics, [loss_sum, acc_sum]


def test(config):
    with open(config.word_emb_file, "r", encoding="utf-8") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r", encoding="utf-8") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    total = meta["total"]

    print("Loading model...")
    test_batch = get_dataset(config.test_record_file, get_record_parser(
        config, is_test=True), config).make_one_shot_iterator()

    model = Model(config, test_batch, word_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        answer_dict = {}
        remapped_dict = {}
        for _ in tqdm(range(total // config.batch_size + 1)):
            qa_id, loss, yp = sess.run(
                [model.qa_id, model.loss, model.yp])
            remapped_dict_, answer_dict_ = convert_tokens(
                eval_file, qa_id.tolist(), yp.tolist())
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)

        f = open(config.answer_file, "w", encoding="utf-8")
        for key in answer_dict:
            f.write(str(key)+"\t"+answer_dict[key]+"\n")
        # 处理不合法（被丢弃）的测试样本
        # 直接选第一个答案
        ans_list = list(answer_dict.keys())
        with open(config.test_file, "r", encoding="utf-8") as fh:
            for line in fh:
                sample = json.loads(line)
                if sample["query_id"] not in ans_list:
                    f.write(str(sample["query_id"])+"\t"+sample['alternatives'].split("|")[0]+"\n")
        f.close()     