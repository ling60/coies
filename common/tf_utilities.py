import tensorflow as tf
import numpy as np


# returns the nearest key(s) of a ndarray dict, given a ndarray
def similar_by_ndarray(ndarray_to_compare, ndarray_dict, topn=1):
    array_len = len(ndarray_dict)
    assert topn <= array_len
    # populate to the same size of ndarray_dict
    # print(ndarray_to_compare.shape)
    ndarray_to_compare = np.resize(ndarray_to_compare, (array_len,) + ndarray_to_compare.shape)
    tf_values_to_compare = tf.convert_to_tensor(ndarray_to_compare)
    keys, values = zip(*ndarray_dict.items())

    values = [v[None, :] for v in values]
    np_values = np.concatenate(values)

    tf_values = tf.convert_to_tensor(np_values)
    # print(tf_values_to_compare.shape)
    # print(tf_values.shape)
    assert tf_values.shape == tf_values_to_compare.shape
    assert tf.assert_rank(tf_values, 2)
    tf_values = tf.nn.l2_normalize(tf_values, dim=-1)
    tf_values_to_compare = tf.nn.l2_normalize(tf_values_to_compare, dim=-1)

    tf_cos_loss = tf.losses.cosine_distance(tf_values, tf_values_to_compare, dim=-1,
                                            reduction=tf.losses.Reduction.NONE)
    tf_cos_loss = tf.squeeze(tf.abs(tf.reduce_sum(tf_cos_loss, axis=1)))

    min_values = tf.nn.top_k(tf.negative(tf_cos_loss), k=topn)
    most_similar_dict = {}
    with tf.Session() as sess:
        cos_loss, values = sess.run([tf_cos_loss, min_values])
        indices = values.indices
        for i, v in zip(indices, values.values):
            print(keys[i])
            most_similar_dict[keys[i]] = v
    return most_similar_dict
