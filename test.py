import scipy
from pyclustering.cluster.xmeans import xmeans, splitting_type
import random
import common.utilities as utils
import logging
import text_cleaning.aaer_corpus as aaer


logging.basicConfig(level=logging.INFO)
a = aaer.AAERParserSentences()
a.get_tokens()

