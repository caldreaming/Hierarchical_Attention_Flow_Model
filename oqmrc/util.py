import numpy as np
import tensorflow as tf
# import logging


def get_record_parser(config, is_test=False):
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "alternatives": tf.FixedLenFeature([], tf.string),
                                               "y": tf.FixedLenFeature([], tf.string),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })
        context_idxs = tf.reshape(tf.decode_raw(
            features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(
            features["ques_idxs"], tf.int32), [ques_limit])

        alternatives = tf.reshape(tf.decode_raw(
            features["alternatives"], tf.int32), [3])
        y = tf.reshape(tf.decode_raw(
            features["y"], tf.float32), [3])
        qa_id = features["id"]
        return context_idxs, ques_idxs, alternatives, y, qa_id
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        # 数据分桶处理
        # 默认设置buckets是一个40到361，步长40的等差数列
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, alternatives, y, qa_id):
            # 根据context长度划分数据集
            # context的有效词数（不算尾部补齐的0）
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            buckets_min = [np.iinfo(np.int32).min] + buckets
            buckets_max = buckets + [np.iinfo(np.int32).max]
            # buckets_min[i]<c_len<=buckets_max[i]则第i为取True
            conditions_c = tf.logical_and(
                tf.less(buckets_min, c_len), tf.less_equal(c_len, buckets_max))
            # 取满足上面条件的最小的i，i作为这个样本分到的桶id
            bucket_id = tf.reduce_min(tf.where(conditions_c))
            return bucket_id

        def reduce_func(key, elements):
            # 对于window_size个数据进行分batch
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset


def convert_tokens(eval_file, qa_id, yp):
    remapped_dict = {}
    answer_dict = {}
    for qid, p in zip(qa_id, yp):
        labels = eval_file[str(qid)]["labels"]
        query_id = eval_file[str(qid)]["query_id"]
        # 完全匹配则返回预测1
        # logging.info("select label"+str(np.argmax(labels)))
        remapped_dict[qid] = 1 if np.argmax(labels) == p else 0
        answer_dict[query_id] = eval_file[str(qid)]["alternatives"][p]
    return remapped_dict, answer_dict


def evaluate(remapped_dict):
    total = 0
    right = 0
    for key in remapped_dict:
        right += remapped_dict[key]
        total += 1

    return {"acc": right/total}


def acc(preds, ground_truths):
    total = 0
    right = 0
    for pred, ground_truth in zip(preds, ground_truths):
        if pred == ground_truth:
            right += 1
        total += 1

    return float(right)/total
