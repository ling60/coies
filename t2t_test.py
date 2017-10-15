from t2t_models import text_encoding
import os
import tensorflow as tf
import text_cleaning.example_parsing as ex_parsing
import common.constants as const
import common.file_tools as ft


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

N_GRAMS = 10
test_file_source = ft.get_source_file_by_example_file(const.TEST_FILE)
tokens = ex_parsing.ngrams_from_file(test_file_source, N_GRAMS, tagged=False)
# TRAIN_DIR=$DATA_DIR/train/$PROBLEM/$MODEL-$HPARAMS

t = text_encoding.TextEncoding(tokens[:100], tokens[:99] + [tokens[2]])
t.encode()
