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

import tensorflow as tf

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
    num_decode_batches = (len(sorted_inputs) - 1) // decode_hp.batch_size + 1

    def input_fn():
        input_gen = decoding._decode_batch_input_fn(
            problem_id, num_decode_batches, sorted_inputs, inputs_vocab,
            decode_hp.batch_size, decode_hp.max_input_size)
        gen_fn = decoding.make_input_fn_from_generator(input_gen)
        example = gen_fn()
        return decoding._decode_input_tensor_to_features_dict(example, hparams)


    decodes = []
    result_iter = estimator.predict(input_fn)
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
    hparams = trainer_utils.add_problem_hparams(hparams, FLAGS.problems)
    estimator, _ = trainer_utils.create_experiment_components(
        data_dir=data_dir,
        model_name=FLAGS.model,
        hparams=hparams,
        run_config=trainer_utils.create_run_config(output_dir))

    decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
    decode_hp.add_hparam("shards", FLAGS.decode_shards)
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
