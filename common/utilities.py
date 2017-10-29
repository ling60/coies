import numpy
import math
import collections
import itertools
import logging
import common.constants as const
import operator


def display_logging_info(allow=True):
    if allow:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)


def cosine_distance(u, v):
    return abs(numpy.dot(u, v) / (math.sqrt(numpy.dot(u, u)) * math.sqrt(numpy.dot(v, v))))


# returns indexes of top values from a list
def top_n_from_list(l, n, start_max=True):
    return sorted(range(len(l)), key=lambda i: l[i], reverse=start_max)[:n]


def make_distance_dict(vector, wv_dict):
    distance_dict = {}
    for k, v in wv_dict.items():
        distance_dict[k] = abs(cosine_distance(vector, v))
    return distance_dict


def most_common_items(a_dict, topn=None):
    d = collections.Counter()
    d.update(a_dict)
    return d.most_common(topn)


# returns the nearest key(s) of vector dict, given a vector
def similar_by_vector(vector, vector_dict, topn=1):
    distance_dict = make_distance_dict(vector, vector_dict)
    return most_common_items(distance_dict, topn)


# returns a group/cluster of tuples(item, value), where values are similar to the top value (distance between them is
# smaller than average
def get_top_group(list_of_tuples, distance_threshold=None):
    list_of_tuples = list(list_of_tuples)
    if len(list_of_tuples) <=1:
        return list_of_tuples
    assert type(list_of_tuples[0]) is tuple or type(list_of_tuples[0]) is list
    if distance_threshold:  # filter by distance_threshold
        list_of_tuples[:] = [t for t in list_of_tuples if t[-1] > distance_threshold]
    if len(list_of_tuples) <= 1:
        return list_of_tuples
    sorted_list = sorted(list_of_tuples, key=operator.itemgetter(-1), reverse=True)
    avg_distance = (sorted_list[0][-1] - sorted_list[-1][-1])/(len(sorted_list)-1)
    top_group = [sorted_list[0]]
    for i in range(1, len(sorted_list)):
        if top_group[-1][-1] - sorted_list[i][-1] < avg_distance:
            top_group.append(sorted_list[i])
        else:
            return top_group


def subset_dict_by_list(a_dict, list_of_keys):
    # print("a_dict", a_dict)
    return {k: a_dict[k] for k in list_of_keys if k in a_dict}


# def subset_dict_by_list2(a_dict, list_of_keys):
#     # returns a dict whose keys exist in the strings of list_of_keys
#     str_list_of_keys = iter_to_string(flatten_list(list_of_keys))
#     sub_dict = {}
#     for k in a_dict.keys():
#         if iter_to_string(k) in str_list_of_keys:
#             sub_dict[k] = a_dict[k]
#     return sub_dict
def subset_dict_by_list2(a_dict, list_of_keys):
    # returns a dict whose keys exist in the strings of list_of_keys
    keys = flatten_list(list_of_keys)
    sub_dict = {}
    for k in a_dict.keys():
        if type(k) is str:
            k = spaced_string_to_tuple(k)
        if is_sublist_of(k, keys):
            sub_dict[k] = a_dict[k]
    return sub_dict


def word_vector_to_dict_by_list(wv, list_of_keys):
    # logging.info('word_vector_to_dict_by_list')
    # print(wv, list_of_keys[0])
    if 'vocab' in wv:
        return {k: wv[k] for k in list_of_keys if k in wv.vocab}
    else:
        return {k: wv[k] for k in list_of_keys}


def flatten_list(list2d):
    return list(itertools.chain.from_iterable(list2d))


# generate n-grams from given list
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] ->
# [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11]]
def ngram_from_list(l, n):
    assert type(l) is list
    grams = [l[i:i + n] for i in range(len(l) - n + 1)]
    return grams


# ngrams in a non overlapping order:[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] ->
# [(1, 2, 3), (4, 5, 6), (7, 8, 9), (2, 3, 4), (5, 6, 7), (8, 9, 10), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
def sequenced_ngrams_from_list(l, n):
    assert type(l) is list
    grams = []
    for i in range(n):
        grams += zip(*(iter(l[i:]),) * n)
    return grams


def iter_to_string(it):
    return const.UNIQUE_DELIMITER.join(it)


def list_to_str_line(l):
    return "%s\n" % ' '.join(l)


def string_to_list(s):
    assert type(s) is str
    return s.split(const.UNIQUE_DELIMITER)


# check if a list is a sublist of another
def is_sublist_of(l1, l2):
    l1 = list(l1)
    l2 = list(l2)
    if len(l1) == 0 or len(l2) == 0:
        return False
    l1 = l1 + [''] if l1[-1] else l1
    l2 = l2 + [''] if l2[-1] else l2
    w = iter_to_string(l1)
    t = iter_to_string(l2)
    return w in t


# calculate the result of one tuple plus another: (1, 2) plus (3, 4) = (4, 6)
def tuple_add(xs, ys):
    return tuple(x + y for x, y in zip(xs, ys))


# [[word1,tag1],[word2,tag2]..] => [word1, word2, ...]
def sentence_from_tagged_ngram(tagged_ngram):
    # logging.info(tagged_ngram)
    return [t[0] for t in tagged_ngram]


# sort dict. returns a list of tuples as [(k,v)...] which is sorted by v, incrementally.
def sorted_tuples_from_dict(a_dict):
    return sorted(a_dict.items(), key=lambda x: x[1])


def spaced_string_to_tuple(spaced_str):
    return tuple(spaced_str.split(' '))
