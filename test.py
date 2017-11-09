import scipy
from pyclustering.cluster.xmeans import xmeans, splitting_type
import random
import common.utilities as utils
import logging
import text_cleaning.aaer_corpus as aaer
import text_cleaning.example_parsing as ex_parsing
import common.constants as const
import model_testing.context_based_models as cb


logging.basicConfig(level=logging.INFO)
a = cb.PhraseVecBigrams()
print(a.aaer_model.get_bigrams(['esafetyworld', 'inc']))

