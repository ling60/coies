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

r"""Decode from trained T2T models.

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
from tensorflow.contrib import learn

import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "", "Training directory to load from.")
flags.DEFINE_string("decode_from_file", None, "Path to decode file")
flags.DEFINE_string("decode_to_file", None,
                    "Path prefix to inference output file")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
flags.DEFINE_string("t2t_usr_dir", "",
                    "Path to a Python module that will be imported. The "
                    "__init__.py file should include the necessary imports. "
                    "The imported files should contain registrations, "
                    "e.g. @registry.register_model calls, that will then be "
                    "available to the t2t-decoder.")
flags.DEFINE_string("master", "", "Address of TensorFlow master.")
flags.DEFINE_string("schedule", "train_and_evaluate",
                    "Must be train_and_evaluate for decoding.")


def numpy_input_fn(x,
                   y=None,
                   batch_size=128,
                   num_epochs=1,
                   shuffle=None,
                   queue_capacity=1000,
                   num_threads=1):
  """Returns input function that would feed dict of numpy arrays into the model.

  This returns a function outputting `features` and `target` based on the dict
  of numpy arrays. The dict `features` has the same keys as the `x`.

  Example:

  ```python
  age = np.arange(4) * 1.0
  height = np.arange(32, 36)
  x = {'age': age, 'height': height}
  y = np.arange(-32, -28)

  with tf.Session() as session:
    input_fn = numpy_io.numpy_input_fn(
        x, y, batch_size=2, shuffle=False, num_epochs=1)
  ```

  Args:
    x: dict of numpy array object.
    y: numpy array object. `None` if absent.
    batch_size: Integer, size of batches to return.
    num_epochs: Integer, number of epochs to iterate over data. If `None` will
      run forever.
    shuffle: Boolean, if True shuffles the queue. Avoid shuffle at prediction
      time.
    queue_capacity: Integer, size of queue to accumulate.
    num_threads: Integer, number of threads used for reading and enqueueing. In
      order to have predicted and repeatable order of reading and enqueueing,
      such as in prediction and evaluation mode, `num_threads` should be 1.

  Returns:
    Function, that has signature of ()->(dict of `features`, `target`)

  Raises:
    ValueError: if the shape of `y` mismatches the shape of values in `x` (i.e.,
      values in `x` have same shape).
    TypeError: `x` is not a dict or `shuffle` is not bool.
  """

  if not isinstance(shuffle, bool):
    raise TypeError('shuffle must be explicitly set as boolean; '
                    'got {}'.format(shuffle))

  def input_fn():
    """Numpy input function."""
    if not isinstance(x, dict):
      raise TypeError('x must be dict; got {}'.format(type(x).__name__))

    # Make a shadow copy and also ensure the order of iteration is consistent.
    ordered_dict_x = collections.OrderedDict(
        sorted(x.items(), key=lambda t: t[0]))

    unique_target_key = _get_unique_target_key(ordered_dict_x)
    if y is not None:
      ordered_dict_x[unique_target_key] = y

    if len(set(v.shape[0] for v in ordered_dict_x.values())) != 1:
      shape_dict_of_x = {k: ordered_dict_x[k].shape
                         for k in ordered_dict_x.keys()}
      shape_of_y = None if y is None else y.shape
      raise ValueError('Length of tensors in x and y is mismatched. All '
                       'elements in x and y must have the same length.\n'
                       'Shapes in x: {}\n'
                       'Shape for y: {}\n'.format(shape_dict_of_x, shape_of_y))

    queue = feeding_functions._enqueue_data(  # pylint: disable=protected-access
        ordered_dict_x,
        queue_capacity,
        shuffle=shuffle,
        num_threads=num_threads,
        enqueue_size=batch_size,
        num_epochs=num_epochs)

    features = (queue.dequeue_many(batch_size) if num_epochs is None
                else queue.dequeue_up_to(batch_size))

    # Remove the first `Tensor` in `features`, which is the row number.
    if len(features) > 0:
      features.pop(0)

    features = dict(zip(ordered_dict_x.keys(), features))
    if y is not None:
      target = features.pop(unique_target_key)
      return features, target
    return features

  return input_fn


def _decode_batch_input_fn(problem_id, num_decode_batches, sorted_inputs,
                           vocabulary, batch_size, max_input_size):
    tf.logging.info(" batch %d" % num_decode_batches)
    # First reverse all the input sentences so that if you're going to get OOMs,
    # you'll see it in the first batch
    sorted_inputs.reverse()
    for b in range(num_decode_batches):
        tf.logging.info("Decoding batch %d" % b)
        batch_length = 0
        batch_inputs = []
        for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
            tf.logging.info(inputs)
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


def _decode_input_tensor_to_features_dict(feature_map, hparams):
    """Convert the interactive input format (see above) to a dictionary.

  Args:
    feature_map: a dictionary with keys `problem_choice` and `input` containing
      Tensors.
    hparams: model hyperparameters

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
    # y = tf.constant([4, 466,   7, 320,   3,   1,   0,   0])
    return features, x


def decode_from_file(estimator, filename, decode_hp, decode_to_file=None):
    """Compute predictions on entries in filename and write them out."""
    if not decode_hp.batch_size:
        decode_hp.batch_size = 32
        tf.logging.info(
            "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)

    hparams = estimator.params
    problem_id = decode_hp.problem_idx
    inputs_vocab = hparams.problems[problem_id].vocabulary["inputs"]
    targets_vocab = hparams.problems[problem_id].vocabulary["targets"]
    problem_name = FLAGS.problems.split("-")[problem_id]
    tf.logging.info("Performing decoding from a file.")
    sorted_inputs, sorted_keys = decoding._get_sorted_inputs(filename, decode_hp.shards)
    # print(sorted_inputs)
    num_decode_batches = (len(sorted_inputs) - 1) // decode_hp.batch_size + 1

    def input_fn1():
        # encoded_inputs = []
        # for ngram in sorted_inputs:
        #     encoded_inputs.append(inputs_vocab.encode(ngram))
        # x = tf.convert_to_tensor(encoded_inputs)
        input_gen = _decode_batch_input_fn(
            problem_id, num_decode_batches, sorted_inputs, inputs_vocab,
            decode_hp.batch_size, decode_hp.max_input_size)
        gen_fn = decoding.make_input_fn_from_generator(input_gen)
        feature_map = gen_fn()
        x = tf.convert_to_tensor(feature_map["inputs"])
        x = tf.expand_dims(x, axis=[2])
        x = tf.expand_dims(x, axis=3)
        x = tf.to_int32(x)
        features = {"inputs": x}
        # features["targets"] = tf.expand_dims(x, axis=[3])
        p_hparams = hparams.problems[problem_id]
        features["problem_choice"] = np.array(problem_id).astype(np.int32)
        features["input_space_id"] = tf.constant(p_hparams.input_space_id)
        features["target_space_id"] = tf.constant(p_hparams.target_space_id)
        features["decode_length"] = tf.shape(x)[1] + 50

        return features, x

    def input_fn2():
        encoded_inputs = []
        for ngram in sorted_inputs:
            encoded_inputs.append(inputs_vocab.encode(ngram))
        x = np.array(encoded_inputs).astype(np.int32)

        x = np.expand_dims(x, axis=2)
        x = np.expand_dims(x, axis=3)
        features = {"inputs": x}
        # features["targets"] = tf.expand_dims(x, axis=[3])
        p_hparams = hparams.problems[problem_id]
        features["problem_choice"] = np.array([problem_id]).astype(np.int32)
        features["input_space_id"] = np.array([p_hparams.input_space_id]).astype(np.int32)
        features["target_space_id"] = np.array([p_hparams.target_space_id]).astype(np.int32)
        features["decode_length"] = np.array([x.shape[1] + 50]).astype(np.int32)

        # print(features)
        # for k, v in features.items():
        #     print(type(v))
        #     # print(v.shape[0])
        #     print(set(v.shape)[0])
        #     # print(len(set(v.shape[0])))

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=x,
            y=x,
            batch_size=decode_hp.batch_size,
            num_epochs=1,
            shuffle=False)
        return eval_input_fn

    def eval_inputs():
        """Returns training set as Operations.
        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        encoded_inputs = []
        for ngram in sorted_inputs:
            encoded_inputs.append(inputs_vocab.encode(ngram))
        tf_inputs = tf.convert_to_tensor(encoded_inputs)

        # Define placeholders
        # images_placeholder = tf.placeholder(
        #     images.dtype, images.shape)
        # labels_placeholder = tf.placeholder(
        #     labels.dtype, labels.shape)
        # Build dataset iterator
        dataset = tf.contrib.data.Dataset.from_tensor_slices(
            tf_inputs)
        # dataset = dataset.repeat(None)  # Infinite iterations
        # dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(decode_hp.batch_size)
        iterator = dataset.make_one_shot_iterator()
        print(iterator)
        x = iterator.get_next()
        x = tf.expand_dims(x, axis=2)
        x = tf.expand_dims(x, axis=3)
        features = {"inputs": x}
        # features["targets"] = tf.expand_dims(x, axis=[3])
        p_hparams = hparams.problems[problem_id]
        features["problem_choice"] = np.array(problem_id).astype(np.int32)
        features["input_space_id"] = tf.constant(p_hparams.input_space_id)
        features["target_space_id"] = tf.constant(p_hparams.target_space_id)
        features["decode_length"] = tf.shape(x)[1] + 50
        # Return batched (features, labels)
        return features, x

    def input_fn():
        input_gen = _decode_batch_input_fn(
            problem_id, num_decode_batches, sorted_inputs, inputs_vocab,
            decode_hp.batch_size, decode_hp.max_input_size)
        gen_fn = decoding.make_input_fn_from_generator(input_gen)
        example = gen_fn()
        return _decode_input_tensor_to_features_dict(example, hparams)
        # features['inputs'] = tf.expand_dims(features['inputs'], axis=3)
        # return features
        # return _decode_input_tensor_to_features_dict(example, hparams)

    decodes = []
    p_hparams = hparams.problems[problem_id]
    print(p_hparams.target_modality)
    my_hook = my_hooks.PredictHook()   # (tensors=[], every_n_iter=30)
    # my_hook.begin()
    # result_iter = estimator.predict(input_fn)
    result_iter = estimator.evaluate(input_fn1, hooks=[my_hook])
    # result_iter = []
    # experiment = learn.Experiment(
    #     estimator=estimator,
    #     train_input_fn=None,
    #     eval_input_fn=input_fn,
    #     eval_hooks=[my_hook],
    #     eval_delay_secs=None,
    #     eval_steps=None,
    #     continuous_eval_throttle_secs=0
    # )
    #
    # def eval_output_fn(eval_result):
    #     tf.logging.info(eval_result)
    #     # print(estimator._eval_metrics)
    #     # tf_eval = tf.Variable("", trainable=False, name='eval')
    #     if type(eval_result) is dict and len(eval_result) < 2:
    #         return False
    #         # raise EnvironmentError(eval_result)
    #         # tf_eval = tf.Variable(eval_result, trainable=False, name='eval')
    #         # tf.logging.info(tf_eval)
    #     return True
    #
    # for i in range(10):
    #     print(experiment.evaluate())
    # experiment.continuous_eval(continuous_eval_predicate_fn=eval_output_fn, evaluate_checkpoint_only_once=False)

    for result in result_iter:
        if decode_hp.return_beams:
            beam_decodes = []
            output_beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
            for k, beam in enumerate(output_beams):
                tf.logging.info("BEAM %d:" % k)
                decoded_outputs, _ = decoding.log_decode_results(result["inputs"], beam,
                                                        problem_name, None,
                                                        inputs_vocab, targets_vocab)
                beam_decodes.append(decoded_outputs)
            decodes.append("\t".join(beam_decodes))
        else:
            decoded_outputs, _ = decoding.log_decode_results(result["inputs"],
                                                    result["outputs"], problem_name,
                                                    None, inputs_vocab, targets_vocab)
            decodes.append(decoded_outputs)

    # Reversing the decoded inputs and outputs because they were reversed in
    # _decode_batch_input_fn
    sorted_inputs.reverse()
    decodes.reverse()
    # Dumping inputs and outputs to file filename.decodes in
    # format result\tinput in the same order as original inputs
    if decode_to_file:
        output_filename = decode_to_file
    else:
        output_filename = filename
    if decode_hp.shards > 1:
        base_filename = output_filename + ("%.2d" % FLAGS.worker_id)
    else:
        base_filename = output_filename
    decode_filename = decoding._decode_filename(base_filename, problem_name, decode_hp)
    tf.logging.info("Writing decodes into %s" % decode_filename)
    outfile = tf.gfile.Open(decode_filename, "w")
    for index in range(len(sorted_inputs)):
        outfile.write("%s\n" % (decodes[sorted_keys[index]]))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
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

    estimator, func_dict = trainer_utils.create_experiment_components(
        data_dir=data_dir,
        model_name=FLAGS.model,
        hparams=hparams,
        run_config=trainer_utils.create_run_config(output_dir))

    decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
    decode_hp.add_hparam("shards", FLAGS.decode_shards)
    decode_hp.batch_size = 2

    # eval_input_fn = func_dict[tf.estimator.ModeKeys.EVAL]
    # estimator.evaluate(input_fn=eval_input_fn, hooks=[my_hooks.PredictHook()])
    #
    # return

    if FLAGS.decode_interactive:
        decoding.decode_interactively(estimator, decode_hp)
    elif FLAGS.decode_from_file:
        decode_from_file(estimator, FLAGS.decode_from_file, decode_hp,
                         FLAGS.decode_to_file)
    else:
        decoding.decode_from_dataset(estimator,
                                     FLAGS.problems.split("-"), decode_hp,
                                     FLAGS.decode_to_file)


if __name__ == "__main__":
    tf.app.run()
