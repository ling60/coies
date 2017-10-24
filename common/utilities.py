import numpy
import math
import collections
import itertools
import logging
import common.constants as const


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


# returns the nearest key(s) of vector dict, given a vector
def similar_by_vector(vector, vector_dict, topn=1):
    if topn <= len(vector_dict):
        distance_dict = {}
        for k, v in vector_dict.items():
            distance_dict[k] = abs(cosine_distance(vector, v))
            # print(distance_dict[k])
            # print(type(vector_dict[k]))
        d = collections.Counter()
        # print(vector_dict)
        d.update(distance_dict)
        return d.most_common(topn)
    else:
        return vector_dict


def subset_dict_by_list(a_dict, list_of_keys):
    # print("a_dict", a_dict)
    return {k: a_dict[k] for k in list_of_keys if k in a_dict}


def subset_dict_by_list2(a_dict, list_of_keys):
    # returns a dict whose keys exist in the strings of list_of_keys
    str_list_of_keys = iter_to_string(flatten_list(list_of_keys))
    sub_dict = {}
    for k in a_dict.keys():
        if iter_to_string(k) in str_list_of_keys:
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
    return sorted(a_dict.items(), key=lambda x:x[1])
