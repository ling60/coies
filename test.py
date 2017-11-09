import scipy
from pyclustering.cluster.xmeans import xmeans, splitting_type
import random
import common.utilities as utils
import logging
import text_cleaning.aaer_corpus as aaer
import text_cleaning.example_parsing as ex_parsing
import common.constants as const


logging.basicConfig(level=logging.INFO)
# a = aaer.AAERParserPhrases()
# print(a.get_tokens())
# sentences = ex_parsing.sentences_from_file(const.EXAMPLE_FILE)
# print(sentences)
# print(list(a.get_trigrams(sentences)))
print(ex_parsing.one_to_n_grams_from_file(const.EXAMPLE_FILE))
