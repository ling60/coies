# this file contains word2vec and fast text methods of computing word embeddings. The resulting embeddings will be
# context independent, or bag of embeddings, however.

import gensim
from gensim.models.wrappers import FastText as gensimFastText
import fasttext
import text_cleaning.example_parsing_no_position as ex_parsing_np
import common.constants as const
import common.file_tools as ft
import common.utilities as util
import logging
import os


def word_vectors_from_file(file_name):
    tokens, entity_dict = ex_parsing_np.parse_file(os.path.join(const.DATA_PATH, file_name))
    model = gensim.models.Word2Vec(tokens, min_count=1)
    return model, entity_dict


def remove_punctuations_from_entity_dict(entity_dict):
    for k, value_list in entity_dict.items():
        value_list[:] = [ft.text_tokenizer(value) for value in value_list]
        entity_dict[k] = value_list
    return entity_dict


def sentences_from_file_list(file_path_list):
    total_tokens = []
    for file in file_path_list:
        tokens, _ = ex_parsing_np.parse_file(file)
        # print(tokens)
        total_tokens.extend(tokens)
    return total_tokens


# a customised version of gensim.models.Word2Vec. Mainly, setting min_count to 1
def word2vec(tokens, min_count=1, size=300, alpha=0.025, window=5,
             sample=0.001, workers=10, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1,
             iteration=5, batch_words=10000):
    if type(tokens[0]) is not list:
        tokens = [tokens]
    print('word2vec: ', tokens[0])
    model = gensim.models.Word2Vec(tokens, min_count=min_count, size=size, alpha=alpha, window=window,
                                   sample=sample, workers=workers, min_alpha=min_alpha, sg=sg, hs=hs, negative=negative,
                                   cbow_mean=cbow_mean, iter=iteration, batch_words=batch_words)
    return model


# returns a fast text model with fasttext package
def fasttext_model_from_file(file_path):
    save_file_name = os.path.join(const.GENERATED_DATA_DIR, const.FASTTEXT_PREFIX + file_path.split('/')[-1])
    try:
        model = fasttext.load_model(save_file_name + '.bin', encoding='utf-8')
        logging.info('model loaded:' + save_file_name)
    except ValueError:
        model = fasttext.cbow(file_path, const.FASTTEXT_PREFIX + file_path.split('/')[-1],
                              encoding='utf-8', min_count=1, lr=0.1)
    return model


# using gensim wrapper instead
def fasttext_model_from_file2(file_path):
    save_file_name = os.path.join(const.GENERATED_DATA_DIR, const.FASTTEXT_PREFIX + file_path.split('/')[-1])
    try:
        model = gensimFastText.load_fasttext_format(save_file_name + '.bin', encoding='utf-8')
        logging.info('model loaded:' + save_file_name)
    except FileNotFoundError:
        fastext_bin_path = os.path.join(const.ROOT_DIR, 'fasttext/fastText')
        model = gensimFastText.train(fastext_bin_path, file_path, min_count=1)
    return model.wv


def make_vec_file_from_wiki_model(sentences, wiki_aaer_vec_name):
    flatten_tokens = util.flatten_list(sentences)
    ft.filter_vec_file_by_set(const.FASTTEXT_WIKI_PATH, set(flatten_tokens), wiki_aaer_vec_name)
