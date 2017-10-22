"""Data generators for AAER corpus."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

import tensorflow as tf

import common.constants as const
import text_cleaning.aaer_corpus as aaer

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


class TextGenerateProblem(problem.Text2TextProblem):
    """Base class for text generating problems."""

    @property
    def is_character_level(self):
        return False

    @property
    def num_shards(self):
        return 100

    @property
    def vocab_name(self):
        return "vocab.aaer"

    @property
    def use_subword_tokenizer(self):
        return True


def generator_from_ngrams(ngrams, token_vocab, target_size=None, window_size=2, eos=None):
    """generator from a list of ngrams.
  Args:
    ngrams: list of n_grams.
    window_size: time shifting distance between inputs and targets
    target_size: the length of target line. should be smaller than length of ngram
    token_vocab: text_encoder.TextEncoder object.
    eos: integer to append at the end of each sequence (default: None).
  Returns:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from ngrams.
  """
    n = len(ngrams[0])
    if target_size is not None:
        assert target_size < n
    else:
        target_size = n

    eos_list = [] if eos is None else [eos]
    epoch_size = len(ngrams) - window_size
    for i in range(epoch_size):
        source_ints = token_vocab.encode(ngrams[i]) + eos_list
        target_ints = token_vocab.encode(ngrams[i + window_size][-target_size:]) + eos_list
        yield {"inputs": source_ints, "targets": target_ints}


def token_generator(source_path, target_path, token_vocab, eos=None):
    """Generator for sequence-to-sequence tasks that uses tokens.

  This generator assumes the files at source_path and target_path have
  the same number of lines and yields dictionaries of "inputs" and "targets"
  where inputs are token ids from the " "-split source (and target, resp.) lines
  converted to integers using the token_map.

  Args:
    source_path: path to the file with source sentences.
    target_path: path to the file with target sentences.
    token_vocab: text_encoder.TextEncoder object.
    eos: integer to append at the end of each sequence (default: None).

  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from tokens in the file lines.
  """
    eos_list = [] if eos is None else [eos]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            while source and target:
                source_ints = token_vocab.encode(source.strip()) + eos_list
                target_ints = token_vocab.encode(target.strip()) + eos_list
                yield {"inputs": source_ints, "targets": target_ints}
                source, target = source_file.readline(), target_file.readline()


@registry.register_problem
class AAERGenerateProblem(TextGenerateProblem):
    """Problem spec for AAER"""

    @property
    def targeted_vocab_size(self):
        return 40000

    @property
    def vocab_name(self):
        return const.T2T_AAER_VOLCAB_NAME

    def feature_encoders(self, data_dir):
        vocab_filename = os.path.join(data_dir, self.vocab_file)
        # encoder = text_encoder.TokenTextEncoder(vocab_filename, replace_oov="UNK")
        encoder = generator_utils.get_or_generate_vocab_inner(data_dir=const.T2T_DATA_DIR,
                                                              vocab_filename=vocab_filename,
                                                              vocab_size=self.targeted_vocab_size,
                                                              generator=aaer.AAERParserTokens().get_tokens())
        return {"inputs": encoder, "targets": encoder}

    def generator(self, data_dir, tmp_dir, train):
        """Instance of token generator for AAER training set."""

        token_path = os.path.join(const.T2T_DATA_DIR, const.T2T_AAER_VOLCAB_NAME)

        with tf.gfile.GFile(token_path, mode="a") as f:
            f.write("UNK\n")  # Add UNK to the vocab.
        token_vocab = text_encoder.SubwordTextEncoder(token_path)

        source_path = const.T2T_AAER_SOURCE_PATH   # if train else const.T2T_AAER_SOURCE_PATH + const.T2T_EVAL_POST_FIX
        targets_path = const.T2T_AAER_TARGETS_PATH  # if train else const.T2T_AAER_TARGETS_PATH + const.T2T_EVAL_POST_FIX

        return token_generator(source_path, targets_path, token_vocab,
                               EOS)

    def eval_metrics(self):
        metrics = super().eval_metrics()
        return metrics

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_BPE_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.DE_BPE_TOK
