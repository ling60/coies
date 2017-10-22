from tensor2tensor.data_generators import generator_utils

from text_cleaning import aaer_corpus
import common.constants as const
import text_cleaning.example_parsing as ex_parsing
import common.utilities as util
import common.file_tools as ft

N_GRAMS = 20
TARGET_SIZE = 6
WINDOW_SIZE = 2


def t2t_files_producer(n_grams,
                       source_path,
                       targets_path,
                       target_size=None,
                       window_size=WINDOW_SIZE):
    """produce data files required by tensor2tensor source_path and target_path have
    the same number of lines
    Args:
      n_grams: a list of fix_sized ngrams
      source_path: file for inputs
      targets_path: file for targets
      window_size: time shifting distance between inputs and targets
      target_size: the length of target line. should be smaller than length of ngram
    """
    assert type(n_grams[0]) is list
    n = len(n_grams[0])
    if target_size is not None:
        assert target_size <= n
    else:
        target_size = n

    epoch_size = len(n_grams) - window_size
    with open(source_path, 'w') as f_source:
        with open(targets_path, 'w') as f_targets:
            for i in range(epoch_size):
                f_source.write(util.list_to_str_line(n_grams[i]))
                f_targets.write(util.list_to_str_line(n_grams[i + window_size][-target_size:]))


def make_t2t_training_files():
    aaer = aaer_corpus.AAERParserNGrams(n=N_GRAMS)
    t2t_files_producer(aaer.get_tokens(), const.T2T_AAER_SOURCE_PATH, const.T2T_AAER_TARGETS_PATH,
                       target_size=TARGET_SIZE)


# def make_t2t_vocal_file():
#     aaer = aaer_corpus.AAERParserTokens()
#     tokens = aaer.get_tokens()
#     with open(definitions.T2T_TEMP_DIR + definitions.T2T_AAER_VOLCAB_NAME + '.40000', 'w') as f:
#         f.write('\n'.join(tokens))


def make_vocal_file():
    aaer = aaer_corpus.AAERParserTokens()
    generator_utils.get_or_generate_vocab_inner(data_dir=const.T2T_DATA_DIR,
                                                              vocab_filename=const.T2T_AAER_VOLCAB_NAME,
                                                              vocab_size=40000,
                                                              generator=aaer.get_tokens())


def make_eval_files(source_file_list, tagged=False):
    n_grams = []
    for path in source_file_list:
        n_grams += ex_parsing.ngrams_from_file(path, N_GRAMS, tagged=tagged)
    t2t_files_producer(n_grams[:100], const.T2T_AAER_SOURCE_PATH+const.T2T_EVAL_POST_FIX,
                       const.T2T_AAER_TARGETS_PATH+const.T2T_EVAL_POST_FIX,
                       TARGET_SIZE)


test_file_source = ft.get_source_file_by_example_file(const.TEST_FILE)
# make_eval_files([test_file_source])
make_t2t_training_files()
# make_vocal_file()
