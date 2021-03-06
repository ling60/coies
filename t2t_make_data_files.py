from tensor2tensor.data_generators import generator_utils

from text_cleaning import aaer_corpus
import common.constants as const
import text_cleaning.example_parsing as ex_parsing
import common.utilities as util
import common.file_tools as ft
import logging
import glob
import os
import json

# todo: extra corpus NOT used for ngrams>=100, due to oversize problem
N_GRAMS = 100
# TARGET_SIZE = 5
WINDOW_SIZE = 20

config_dict = {'ngrams': N_GRAMS,
               'window_size': WINDOW_SIZE}
CONFIG_FILE_NAME = 'my_t2t_config.json'


def save_configs():
    path = os.path.join(const.T2T_DATA_DIR, CONFIG_FILE_NAME)
    with open(path, 'w') as f:
        json.dump(config_dict, f)


def load_configs():
    path = os.path.join(const.T2T_DATA_DIR, CONFIG_FILE_NAME)
    with open(path, 'r') as f:
        conf_dict = json.load(f)
    return conf_dict


def t2t_files_producer(n_grams,
                       source_path,
                       targets_path,
                       target_size=None,
                       window_size=WINDOW_SIZE):
    """produce data files required by tensor2tensor source_path and target_path have
    the same number of lines target = n_grams[i + window_size][-target_size:]
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


# the slicing logic here should be the same as source_ngram_from_target_ngram
def replace_by_window_size(target_ngram, tokens, window_size):
    target_ngram[window_size:-window_size] = tokens
    return target_ngram


def source_ngram_from_target_ngram(target_ngram, window_size):
    return target_ngram[window_size:-window_size]


def t2t_files_producer2(n_grams,
                        source_path,
                        targets_path,
                        window_size=WINDOW_SIZE):
    """produce data files required by tensor2tensor source_path and target_path have
    the same number of lines. target = window + source + window
    Args:
      n_grams: a list of fix_sized ngrams
      source_path: file for inputs
      targets_path: file for targets
      window_size: time shifting distance between inputs and targets
    """
    assert type(n_grams[0]) is list or type(n_grams[0]) is tuple
    epoch_size = len(n_grams) - window_size
    with open(source_path, 'w') as f_source:
        with open(targets_path, 'w') as f_targets:
            for i in range(epoch_size):
                f_targets.write(util.list_to_str_line(n_grams[i]))
                f_source.write(util.list_to_str_line(source_ngram_from_target_ngram(n_grams[i], window_size)))


def get_target_gram_n(source_gram_n, window_size):
    return source_gram_n + window_size * 2


def make_t2t_training_files(ngram_min=1, ngram_max=N_GRAMS):
    print("saving configs..")
    save_configs()
    print("making training files..")
    m = get_target_gram_n(ngram_min, WINDOW_SIZE)
    n = get_target_gram_n(ngram_max, WINDOW_SIZE)
    if m == n:
        aaer = aaer_corpus.AAERParserNGrams(n=n)
    else:
        aaer = aaer_corpus.AAERExParserM2NGrams(m=m, n=n)
    # t2t_files_producer(aaer.get_tokens(), const.T2T_AAER_SOURCE_PATH, const.T2T_AAER_TARGETS_PATH,
    #                    target_size=TARGET_SIZE)
    t2t_files_producer2(aaer.get_tokens(enable_save=False), const.T2T_AAER_SOURCE_PATH, const.T2T_AAER_TARGETS_PATH)


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
    # t2t_files_producer(n_grams[:100], const.T2T_AAER_SOURCE_PATH + const.T2T_EVAL_POST_FIX,
    #                    const.T2T_AAER_TARGETS_PATH + const.T2T_EVAL_POST_FIX,
    #                    TARGET_SIZE)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # todo:Important! delete save files generated by testing last model!
    files_to_del = glob.glob(const.GENERATED_DATA_DIR + '/' + const.DL_DOC_DICT_PREFIX + '*')
    print('deleting following data files:')
    for f in files_to_del:
        print(f)
        os.remove(f)
    # make_eval_files([test_file_source])
    make_t2t_training_files(N_GRAMS, N_GRAMS)
    make_vocal_file()

