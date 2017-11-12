#!/usr/bin/env python
# coding=utf-8

"""Encode text (ngrams) to doc embeddings

IMPORTANT: Models should be trained first!!!!
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_utils
from tensor2tensor.utils import usr_dir

from t2t_models import my_hooks
import common.constants as const

import tensorflow as tf

import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS
output_dir = os.path.join(const.T2T_DATA_DIR, 'train', const.T2T_PROBLEM, const.T2T_MODEL + '-' + const.T2T_HPARAMS)

flags.DEFINE_string("output_dir", output_dir, "Training directory to load from.")
flags.DEFINE_string("decode_from_file", None, "Path to decode file")
flags.DEFINE_string("decode_to_file", None,
                    "Path prefix to inference output file")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
flags.DEFINE_string("t2t_usr_dir", const.T2T_USER_DIR,
                    "Path to a Python module that will be imported. The "
                    "__init__.py file should include the necessary imports. "
                    "The imported files should contain registrations, "
                    "e.g. @registry.register_model calls, that will then be "
                    "available to the t2t-decoder.")
flags.DEFINE_string("master", "", "Address of TensorFlow master.")
flags.DEFINE_string("schedule", "train_and_evaluate",
                    "Must be train_and_evaluate for decoding.")

FLAGS.problems = const.T2T_PROBLEM
FLAGS.model = const.T2T_MODEL
FLAGS.hparams_set = const.T2T_HPARAMS
FLAGS.hparams = None
FLAGS.data_dir = const.T2T_DATA_DIR

""" encode given text (tokens) inputs into embeddings by t2t model. 
    IMPORTANT: Models should be trained first!!!!
"""


class TextEncoding:
    def __init__(self, str_tokens, eval_tokens=None, batch_size=1000):
        """
        Args:
            batch_size: used for encoding
            str_tokens: the original token inputs, as the format of ['t1', 't2'...]. The items within should be strings
            eval_tokens: if not None, then should be the same length as tokens, for similarity comparisons.
        """
        assert type(str_tokens) is list
        assert len(str_tokens) > 0
        assert type(str_tokens[0]) is str
        self.str_tokens = str_tokens
        if eval_tokens is not None:
            assert (len(eval_tokens) == len(str_tokens) and type(eval_tokens[0]) is str)
        self.eval_tokens = eval_tokens
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.logging.info('tf logging set to INFO by: %s' % self.__class__.__name__)

        usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
        trainer_utils.log_registry()
        trainer_utils.validate_flags()
        assert FLAGS.schedule == "train_and_evaluate"
        data_dir = os.path.expanduser(FLAGS.data_dir)
        out_dir = os.path.expanduser(FLAGS.output_dir)

        hparams = trainer_utils.create_hparams(
            FLAGS.hparams_set, data_dir, passed_hparams=FLAGS.hparams)

        trainer_utils.add_problem_hparams(hparams, FLAGS.problems)
        # print(hparams)
        hparams.eval_use_test_set = True

        self.estimator, _ = trainer_utils.create_experiment_components(
            data_dir=data_dir,
            model_name=FLAGS.model,
            hparams=hparams,
            run_config=trainer_utils.create_run_config(out_dir))

        decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
        decode_hp.add_hparam("shards", FLAGS.decode_shards)
        decode_hp.batch_size = batch_size
        self.decode_hp = decode_hp
        self.arr_results = None
        self._encoding_len = 1

    def encode(self, encoding_len=None):
        if encoding_len:
            self._encoding_len = encoding_len
        else:
            encoding_len = self._encoding_len
        estimator = self.estimator
        decode_hp = self.decode_hp
        hparams = estimator.params
        problem_id = decode_hp.problem_idx
        inputs_vocab = hparams.problems[problem_id].vocabulary["inputs"]

        # if eval_tokens exists, add to the str_tokens
        str_tokens = self.str_tokens + self.eval_tokens if self.eval_tokens else self.str_tokens

        tokens_length = len(str_tokens)
        tf.logging.info('token length: %d' % tokens_length)

        # print(str_tokens)
        num_decode_batches = (len(str_tokens) - 1) // decode_hp.batch_size + 1

        def input_fn():
            input_gen = self._decode_batch_input_fn(
                problem_id, num_decode_batches, str_tokens, inputs_vocab,
                decode_hp.batch_size, decode_hp.max_input_size)
            gen_fn = decoding.make_input_fn_from_generator(input_gen)
            example = gen_fn()
            return self._decode_input_tensor_to_features_dict(example, hparams, encoding_len=encoding_len)

        def eval_inputs():
            """Returns training set as Operations.
            Returns:
                (features, labels) Operations that iterate over the dataset
                on every evaluation
            """
            encoded_inputs = []
            for ngram in str_tokens:
                # print(ngram)
                encoded_inputs.append(inputs_vocab.encode(' '.join(ngram)))
                # print(encoded_inputs[-1])
                # print(inputs_vocab.decode(encoded_inputs[-1]))
            tf_inputs = tf.convert_to_tensor(encoded_inputs)

            dataset = tf.contrib.data.Dataset.from_tensor_slices(
                tf_inputs)

            dataset = dataset.batch(decode_hp.batch_size)
            iterator = dataset.make_one_shot_iterator()

            x = iterator.get_next()
            # required by t2t models
            x = tf.expand_dims(x, axis=2)
            x = tf.expand_dims(x, axis=3)
            # y is just a 'place holder' here, as required by evaluate process
            y = x[:, 0:encoding_len, :, :]

            features = {"inputs": x}
            p_hparams = hparams.problems[problem_id]
            features["problem_choice"] = np.array(problem_id).astype(np.int32)
            features["input_space_id"] = tf.constant(p_hparams.input_space_id)
            features["target_space_id"] = tf.constant(p_hparams.target_space_id)
            features["decode_length"] = tf.shape(x)[1] + 50

            # Return batched (features, labels)
            return features, y

        # embeddings_hook = my_hooks.EmbeddingsHook()
        # _ = estimator.evaluate(input_fn, hooks=[embeddings_hook])
        self.arr_results = self.run_estimator(estimator, input_fn)
        # print(self.arr_results)
        return self.arr_results

    @staticmethod
    def run_estimator(estimator, input_fn):
        embeddings_hook = my_hooks.EmbeddingsHook()
        _ = estimator.evaluate(input_fn, hooks=[embeddings_hook])
        arr_results = np.squeeze(np.concatenate(embeddings_hook.embeddings, axis=0), axis=1)
        return arr_results

    def top_n_similarity(self, top_n=3, arr_results=None):
        # print(arr_results)
        if self.eval_tokens:
            eval_tokens_length = len(self.eval_tokens)
            arr_embeddings = arr_results[:eval_tokens_length]
            eval_arr_embeddings = arr_results[eval_tokens_length:]
            # print(arr_embeddings-eval_arr_embeddings)
            tf_embeddings = tf.nn.l2_normalize(tf.convert_to_tensor(arr_embeddings), dim=2)
            tf_eval_embeddings = tf.nn.l2_normalize(tf.convert_to_tensor(eval_arr_embeddings), dim=2)
            tf_cos_loss = tf.losses.cosine_distance(tf_embeddings, tf_eval_embeddings, dim=2,
                                                    reduction=tf.losses.Reduction.NONE)
            tf_cos_loss = tf.squeeze(tf.abs(tf.reduce_sum(tf_cos_loss, axis=1)))

            # min_index = tf.argmin(tf_cos_loss)
            min_values = tf.nn.top_k(tf.negative(tf_cos_loss), k=top_n)
            # tf_cos_loss = tf.losses.absolute_difference(tf_embeddings, tf_eval_embeddings)
            with tf.Session() as sess:
                cos_loss, values = sess.run([tf_cos_loss, min_values])
                print(cos_loss)
                print(cos_loss.shape)
                print(values)
                indices = values.indices
                for i in indices:
                    print(self.str_tokens[i], self.eval_tokens[i])

    @staticmethod
    def _decode_input_tensor_to_features_dict(feature_map, hparams, encoding_len=1):
        """Convert the interactive input format (see above) to a dictionary.

      Args:
        feature_map: a dictionary with keys `problem_choice` and `input` containing
          Tensors.
        hparams: model hyperparameters
        encoding_len: the embedding steps. usually 1

      Returns:
        a features dictionary, as expected by the decoder.
      """
        inputs = tf.convert_to_tensor(feature_map["inputs"])
        input_is_image = False

        def input_fn(problem_choice, x=inputs):  # pylint: disable=missing-docstring
            p_hparams = hparams.problems[problem_choice]
            # Add a third empty dimension dimension
            x = tf.expand_dims(x, axis=2)
            x = tf.to_int32(x)
            return (tf.constant(p_hparams.input_space_id), tf.constant(
                p_hparams.target_space_id), x)

        input_space_id, target_space_id, x = decoding.input_fn_builder.cond_on_index(
            input_fn, feature_map["problem_choice"], len(hparams.problems) - 1)

        features = {}
        features["problem_choice"] = feature_map["problem_choice"]
        features["input_space_id"] = input_space_id
        features["target_space_id"] = target_space_id
        features["decode_length"] = (decoding.IMAGE_DECODE_LENGTH
                                     if input_is_image else tf.shape(x)[1] + 50)
        # features["inputs"] = x
        # for evaluation, x needs to be added with a fourth dim. (not needed for prediction)
        x = tf.expand_dims(x, axis=3)
        features["inputs"] = x
        # features["targets"] = tf.fill([5, 1, 1, 1], 0)
        y = x[:, 0:encoding_len, :, :]
        return features, y

    @staticmethod
    def _decode_batch_input_fn(problem_id, num_decode_batches, sorted_inputs,
                               vocabulary, batch_size, max_input_size):
        tf.logging.info(" batch %d" % num_decode_batches)
        # First reverse all the input sentences so that if you're going to get OOMs,
        # you'll see it in the first batch
        # sorted_inputs.reverse()
        for b in range(num_decode_batches):
            tf.logging.info("Decoding batch %d" % b)
            batch_length = 0
            batch_inputs = []
            for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
                input_ids = vocabulary.encode(inputs)
                if max_input_size > 0:
                    # Subtract 1 for the EOS_ID.
                    input_ids = input_ids[:max_input_size - 1]
                input_ids.append(decoding.text_encoder.EOS_ID)
                batch_inputs.append(input_ids)
                if len(input_ids) > batch_length:
                    batch_length = len(input_ids)
            final_batch_inputs = []
            for input_ids in batch_inputs:
                assert len(input_ids) <= batch_length
                x = input_ids + [0] * (batch_length - len(input_ids))
                final_batch_inputs.append(x)

            yield {
                "inputs": np.array(final_batch_inputs).astype(np.int32),
                "problem_choice": np.array(problem_id).astype(np.int32),
            }


class TextSimilarity(TextEncoding):
    def __init__(self, inputs, targets):
        assert len(inputs) == len(targets)
        super().__init__(inputs)
        self.targets = targets

    def encode(self, encoding_len=None):
        if encoding_len:
            self._encoding_len = encoding_len
        else:
            encoding_len = self._encoding_len
        estimator = self.estimator
        decode_hp = self.decode_hp
        hparams = estimator.params
        problem_id = decode_hp.problem_idx
        inputs_vocab = hparams.problems[problem_id].vocabulary["inputs"]
        targets_vocab = hparams.problems[problem_id].vocabulary["targets"]

        str_inputs = self.str_tokens
        str_targets = self.targets

        tokens_length = len(str_inputs)
        tf.logging.info('token length: %d' % tokens_length)

        def eval_inputs():
            """Returns training set as Operations.
            Returns:
                (features, labels) Operations that iterate over the dataset
                on every evaluation
            """
            encoded_inputs = []
            encoded_targets = []
            for ngram in str_inputs:
                # print(ngram)
                encoded_inputs.append(inputs_vocab.encode(ngram))
                # print(encoded_inputs[-1])
                # print(inputs_vocab.decode(encoded_inputs[-1]))
            for s in str_targets:
                encoded_targets.append(targets_vocab.encode(s))

            arr_inputs = np.asarray(encoded_inputs, dtype=np.int32)
            arr_targets = np.asarray(encoded_targets, dtype=np.int32)
            tf_inputs = tf.convert_to_tensor(arr_inputs)
            tf_targets = tf.convert_to_tensor(arr_targets)

            dataset = tf.contrib.data.Dataset.from_tensor_slices(
                (tf_inputs, tf_targets))

            dataset = dataset.batch(decode_hp.batch_size)
            iterator = dataset.make_one_shot_iterator()

            x, y = iterator.get_next()
            # required by t2t models
            x = tf.expand_dims(x, axis=2)
            x = tf.expand_dims(x, axis=3)
            # y is just a 'place holder' here, as required by evaluate process
            y = tf.expand_dims(y, axis=2)
            y = tf.expand_dims(y, axis=3)

            features = {"inputs": x}
            p_hparams = hparams.problems[problem_id]
            features["problem_choice"] = np.array(problem_id).astype(np.int32)
            features["input_space_id"] = tf.constant(p_hparams.input_space_id)
            features["target_space_id"] = tf.constant(p_hparams.target_space_id)
            features["decode_length"] = tf.shape(y)[1]

            # Return batched (features, labels)
            return features, y

        self.arr_results = self.run_estimator(estimator, eval_inputs)
        # print(self.arr_results)
        return self.arr_results

