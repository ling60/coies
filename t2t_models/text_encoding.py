#!/usr/bin/env python
# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decode from trained T2T models.

This binary performs inference using the Estimator API.

Example usage to decode from dataset:

  t2t-decoder \
      --data_dir ~/data \
      --problems=algorithmic_identity_binary40 \
      --model=transformer
      --hparams_set=transformer_base

Set FLAGS.decode_interactive or FLAGS.decode_from_file for alternative decode
sources.
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

import tensorflow as tf

import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS

""" encode given text (tokens) inputs into embeddings by t2t model. 
"""


class TextEncoding:
    def __init__(self, tokens, eval_tokens=None, batch_size=1000):
        """

        Args:
            batch_size: used for encoding
            tokens: the original token inputs, as the format of ['t1', 't2'...]
            eval_tokens: if not None, then should be the same length as tokens, for similarity comparisons.
        """
        assert type(tokens) is list
        assert len(tokens) > 0
        self.tokens = tokens
        if eval_tokens is not None:
            assert len(eval_tokens) == len(tokens)
        self.eval_tokens = eval_tokens
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.logging.info('tf logging set to INFO by: %s' % self.__class__.__name__)

        usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
        trainer_utils.log_registry()
        trainer_utils.validate_flags()
        assert FLAGS.schedule == "train_and_evaluate"
        data_dir = os.path.expanduser(FLAGS.data_dir)
        output_dir = os.path.expanduser(FLAGS.output_dir)

        hparams = trainer_utils.create_hparams(
            FLAGS.hparams_set, data_dir, passed_hparams=FLAGS.hparams)

        trainer_utils.add_problem_hparams(hparams, FLAGS.problems)
        print(hparams)
        hparams.eval_use_test_set = True

        self.estimator, _ = trainer_utils.create_experiment_components(
            data_dir=data_dir,
            model_name=FLAGS.model,
            hparams=hparams,
            run_config=trainer_utils.create_run_config(output_dir))

        decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
        decode_hp.add_hparam("shards", FLAGS.decode_shards)
        decode_hp.batch_size = batch_size
        self.decode_hp = decode_hp

    def encode(self):
        estimator = self.estimator
        decode_hp = self.decode_hp
        hparams = estimator.params
        problem_id = decode_hp.problem_idx
        inputs_vocab = hparams.problems[problem_id].vocabulary["inputs"]

        # if eval_tokens exists, add to the tokens
        tokens = self.tokens + self.eval_tokens if self.eval_tokens else self.tokens

        tokens_length = len(tokens)
        tf.logging.info('token length: %d' % tokens_length)

        # print(tokens)
        # num_decode_batches = (len(tokens) - 1) // decode_hp.batch_size + 1

        def eval_inputs():
            """Returns training set as Operations.
            Returns:
                (features, labels) Operations that iterate over the dataset
                on every evaluation
            """
            encoded_inputs = []
            for ngram in tokens:
                encoded_inputs.append(inputs_vocab.encode(' '.join(ngram)))
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
            y = x[:, 0:3, :, :]

            features = {"inputs": x}
            p_hparams = hparams.problems[problem_id]
            features["problem_choice"] = np.array(problem_id).astype(np.int32)
            features["input_space_id"] = tf.constant(p_hparams.input_space_id)
            features["target_space_id"] = tf.constant(p_hparams.target_space_id)
            features["decode_length"] = tf.shape(x)[1] + 50

            # Return batched (features, labels)
            return features, y

        p_hparams = hparams.problems[problem_id]
        # print(p_hparams.target_modality)
        my_hook = my_hooks.PredictHook()  # (tensors=[], every_n_iter=30)

        # result_iter = estimator.predict(input_fn)
        _ = estimator.evaluate(eval_inputs, hooks=[my_hook])

        arr_results = np.concatenate(my_hook.embeddings, axis=0)
        print(arr_results)
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

            min_index = tf.argmin(tf_cos_loss)
            # tf_cos_loss = tf.losses.absolute_difference(tf_embeddings, tf_eval_embeddings)
            with tf.Session() as sess:
                cos_loss = sess.run([tf_cos_loss, min_index])
                print(cos_loss)
                print(cos_loss.shape)
        else:
            arr_embeddings = np.concatenate(my_hook.embeddings, axis=0)
