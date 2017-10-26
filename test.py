import scipy
from pyclustering.cluster.xmeans import xmeans, splitting_type
import random
import common.utilities as utils

a = [["a", random.random()] for i in range(10)]
print(a)
# xmeans_instance = xmeans(a, criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION, ccore=False)
# print(xmeans_instance.get_clusters())
print(utils.get_top_group(a))

