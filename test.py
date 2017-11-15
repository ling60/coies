import scipy
from pyclustering.cluster.xmeans import xmeans, splitting_type
import random
import common.utilities as utils
import logging
import text_cleaning.aaer_corpus as aaer
import text_cleaning.example_parsing as ex_parsing
import common.constants as const
import model_testing.context_based_models as cb
import t2t_models.text_encoding as t2t_encoding


logging.basicConfig(level=logging.INFO)
head_num = 10
with open(const.T2T_AAER_SOURCE_PATH) as s_f:
    with open(const.T2T_AAER_TARGETS_PATH) as t_f:

        sources = [line.rstrip('\n') for line in utils.file_head(s_f, head_num)]
        targets = [line.rstrip('\n') for line in utils.file_head(t_f, head_num)]
        encoder = t2t_encoding.TextSimilarity(sources, targets)
        print(encoder.encode())
# a = cb.PhraseVecBigrams()
# print(a.aaer_model.get_bigrams(['esafetyworld', 'inc']))
# a = t2t_encoding.TextEncoding(['esafetyworld', 'inc'])
# a.encode()
