# shared constants across this project
import os
import root_path

# file / path constants
ROOT_DIR = root_path.ROOT_DIR
DATA_PATH = os.path.join(os.path.dirname(ROOT_DIR), "data")  # 'D:/work/research/gan-accounting/data'
AAER_PATH = 'sec/aaer'
EX_AAER_PATH = 'sec/admin'  # aaer corpus plus files located in admin dir
EXAMPLE_FILE = os.path.join(DATA_PATH, 'example/34-53330.txt')
TEST_DIR = os.path.join(DATA_PATH, 'test')
TEST_FILE = os.path.join(TEST_DIR, 'comp20547-lagreca.txt')
HUMAN_DIR = os.path.join(DATA_PATH, 'human_results')
GENERATED_DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(GENERATED_DATA_DIR, 'results')

HTML_EXTENSIONS = ["html", "shtml", "htm"]
TEXT_EXTENSIONS = ["txt"]
PICKLE_FILE_EXTENSION = "pkl"

# tensor2tensor related file constants
T2T_DATA_DIR = os.path.join(GENERATED_DATA_DIR, 't2t')
T2T_TEMP_DIR = os.path.join(T2T_DATA_DIR, 'tmp')
T2T_AAER_SOURCE_PATH = os.path.join(T2T_DATA_DIR, 'aaer_source')
T2T_AAER_TARGETS_PATH = os.path.join(T2T_DATA_DIR, 'aaer_targets')
T2T_AAER_VOLCAB_NAME = 'aaer.volcab'
T2T_EVAL_POST_FIX = '_eval'
T2T_PROBLEM = 'aaer_generate_problem'
T2T_MODEL = 'transformer'
T2T_USER_DIR = os.path.join(ROOT_DIR, 't2t_models')
T2T_HPARAMS = 'transformer_base_single_gpu'

# tags related
TAGS = ['comp.', 'date.', 'item.']
TAG_POSTFIX = '.'
TAG_ENDING = '/'
END_TAG = 'end'
IN_TAG = 'in'
NONE_TAG = None
MARK_TAG = 'tag'
UNIQUE_PREFIX = 'LingINGHzdlq'
UNIQUE_DELIMITER = ':'
STARTING_TAGS = TAG_POSTFIX + '>'
ENDING_TAGS = '<' + TAG_ENDING + '>'
REPLACED_STARTING_TAGS = UNIQUE_PREFIX + 'start'
REPLACED_ENDING_TAGS = UNIQUE_PREFIX + 'end'

# model related
FASTTEXT_PREFIX = 'fasttext_'
FASTTEXT_WIKI_PATH = 'D:/work/research/common-data/fasttext/wiki.en.vec'
