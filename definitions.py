import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project Root
GENERATED_DATA_DIR = ROOT_DIR + '/data/'
T2T_DATA_DIR = GENERATED_DATA_DIR + 't2t/'
T2T_TEMP_DIR = T2T_DATA_DIR + 'tmp/'
T2T_AAER_SOURCE_PATH = T2T_DATA_DIR + 'aaer_source'
T2T_AAER_TARGETS_PATH = T2T_DATA_DIR + 'aaer_targets'
T2T_AAER_VOLCAB_NAME = 'aaer.volcab'
