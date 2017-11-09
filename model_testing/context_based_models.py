# this file contains models that take context/position information into account
import text_cleaning.example_parsing as ex_parsing
import gensim
import logging
import common.utilities as utils
import common.constants as const
import common.file_tools as ft
import text_cleaning.aaer_corpus as aaer
import model_testing.word2vec_models as word2vec_models
import numpy as np
import os


# instead of comparing words directly, we will choose most similar context(sentences) first, then find similar words
# from given context.
# method 1: computing doc2vec and then word2vec
# preparing labelled sentences for doc2vec. Sentences should be 2d list
def label_sentences(sentences, label_prefix=None):
    labeled_sentences = []
    for count, sentence in enumerate(sentences):
        # logging.info(sentence)
        label = label_prefix + const.UNIQUE_DELIMITER + str(count) if label_prefix else count
        tagged_doc = gensim.models.doc2vec.TaggedDocument(words=sentence, tags=[label])
        labeled_sentences.append(tagged_doc)
    return labeled_sentences


def label_ngrams_from_file_list(file_path_list, n=10):
    ngrams = []
    for file_path in file_path_list:
        # file_name = ft.file_name_from_path(file_path)
        ngrams += ex_parsing.sequenced_ngrams_from_file(file_path, n)
    labeled_ngrams = label_sentences(ngrams)
    return labeled_ngrams


def find_labeled_ngrams_by_file_name(labeled_ngrams, file_name):
    found_ngrams = []
    for g in labeled_ngrams:
        label_to_find = file_name + const.UNIQUE_DELIMITER
        if label_to_find in g.tags[0]:
            found_ngrams.append(g)
    return found_ngrams


def find_labels_by_sentence(labeled_sentences, sentence):
    assert len(sentence) == len(labeled_sentences[0])
    labels = []
    for labeled_sentence in labeled_sentences:
        if sentence == labeled_sentence.words:
            labels.append(labeled_sentence.tags)
    return labels


# ngrams(sentences) and tags share same labels. The format of label_tags_dict will be:{'label':[[word, tag],[word...]]}
def label_tagged_ngrams(tagged_ngrams, label_prefix=None):
    labeled_ngrams = []
    label_tags_dict = {}
    for count, tagged_ngram in enumerate(tagged_ngrams):
        # logging.info(tagged_ngram)
        ngram = [g[0] for g in tagged_ngram]
        label = label_prefix + str(count) if label_prefix else count
        tagged_doc = gensim.models.doc2vec.TaggedDocument(words=ngram, tags=[label])
        labeled_ngrams.append(tagged_doc)
        label_tags_dict[label] = tagged_ngram
    return labeled_ngrams, label_tags_dict


# find complete ngrams/sentence given tagged words:[[word, tag],..]
def find_ngrams_by_tagged_words(tagged_ngrams, tagged_words):
    ngrams = []
    for tagged_ngram in tagged_ngrams:
        w = filter(None, utils.flatten_list(tagged_words))
        t = filter(None, utils.flatten_list(tagged_ngram))
        if utils.is_sublist_of(w, t):
            ngrams.append(utils.sentence_from_tagged_ngram(tagged_ngram))
    return ngrams


# a wrapper of gensim doc2vec
def doc2vec(labeled_sentences):
    # logging.info(labeled_sentences[0])
    # logging.info(labeled_sentences[0].words)
    # logging.info(labeled_sentences[0].tags)
    # model = gensim.models.doc2vec.Doc2Vec(documents=labeled_sentences, workers=10, dbow_words=0, min_count=1)
    model = gensim.models.doc2vec.Doc2Vec(workers=10, min_count=1, size=512)
    model.build_vocab(labeled_sentences)
    model.train(sentences=labeled_sentences, total_examples=model.corpus_count, epochs=100)
    # logging.info(model.infer_vector(labeled_sentences[0].words))
    return model


def make_doc2vec_model_from_aaer(gram_n=None):
    if gram_n:
        save_fname = os.path.join(const.GENERATED_DATA_DIR, 'aaer_doc2vec_' + str(gram_n) + 'grams')
    else:
        save_fname = os.path.join(const.GENERATED_DATA_DIR, 'aaer_doc2vec_sentences')
    try:
        doc_vec_model = gensim.models.Doc2Vec.load(save_fname)
    except FileNotFoundError:
        logging.info(save_fname + ' not found')
        if gram_n:
            aaer_corpus = aaer.AAERExParserNGrams(n=gram_n)
        else:
            aaer_corpus = aaer.AAERExParserSentences()
        ngrams = aaer_corpus.get_tokens()
        labeled_ngrams = label_sentences(ngrams)
        # labeled_tagged_ngrams = label_ngrams_from_file_list(ft.list_file_paths_under_dir(
        #     os.path.join(const.DATA_PATH, const.AAER_PATH), ['txt']))
        doc_vec_model = doc2vec(labeled_ngrams)
        doc_vec_model.save(save_fname)
    return doc_vec_model


def doc_vector_dict_by_ngrams(doc2vec_model, ngrams):
    assert isinstance(doc2vec_model, gensim.models.Doc2Vec)
    logging.info('ngrams[0]:')
    logging.info(ngrams[0])
    assert type(ngrams[0][0]) is str  # ngrams should be a list of string, not a [str, tag] pair
    doc_vector_dict = {}
    for ngram in ngrams:
        key = const.UNIQUE_DELIMITER.join(ngram)
        # labels = find_labels_by_sentence(tagged_docs, ngram)
        doc_vector_dict[key] = doc2vec_model.infer_vector(ngram)
    return doc_vector_dict


# model2: instead of doc2vec, we will try other shallow transformation of word embeddings
# to calculate distance between ngrams
class DocVecByWordEmbeddings:
    # if not aaer_corpus then docs= will be needed
    def __init__(self, aaer_ex=True, aaer_corpus=False, **kwargs):
        if aaer_ex:  # using aaer extra corpus instead
            self.aaer = True
            self.aaer_model = aaer.AAERExParserSentences()
            self.docs = self.aaer_model.get_tokens()
        elif aaer_corpus:
            self.aaer = True
            self.aaer_model = aaer.AAERParserSentences()
            self.docs = self.aaer_model.get_tokens()
        else:
            self.aaer = False
            self.docs = kwargs['docs']
        self.wv_model = self.wv_training()
        # self.doc_vec_dict = self.make_doc_vec_dict()

    @staticmethod
    def compute_doc_vec(words_vectors):
        raise NotImplementedError

    def wv_training(self):
        if self.aaer:  # aaer corpus is used. So we can save and load
            return self.aaer_model.make_word2vec_model()
        else:
            return word2vec_models.word2vec(self.docs)

    def wv_update(self, docs):
        self.wv_model.build_vocab(docs, update=True)
        self.wv_model.train(docs, total_examples=self.wv_model.corpus_count, epochs=self.wv_model.iter)
        # self.make_doc_vec_dict(update_docs=docs)

    def infer_vector(self, tokens):
        token_vectors = [self.wv_model.wv[token] for token in tokens]
        return self.compute_doc_vec(token_vectors)


class DocVecByWEMean(DocVecByWordEmbeddings):
    @staticmethod
    def compute_doc_vec(words_vectors):
        return np.mean(words_vectors, axis=0)


class DocVecByWESum(DocVecByWEMean):
    @staticmethod
    def compute_doc_vec(words_vectors):
        return np.sum(words_vectors, axis=0)


# this class use gensim phrases input to compute vectors. Note there is "_" as a delimiter for gensim inputs
class PhraseVec:
    def __init__(self):
        self.aaer_model = aaer.AAERExParserPhrases()
        self.wv_model = self.aaer_model.make_word2vec_model()

    def infer_vector(self, tuple_phrase):
        # print(tuple_phrase)
        assert type(tuple_phrase) is tuple or type(tuple_phrase) is list
        token = const.GENSIM_PHRASES_DELIMITER.join(tuple_phrase)
        try:
            vector = self.wv_model[token]
        except KeyError:
            tokens = list(self.aaer_model.get_trigrams(tuple_phrase))
            vectors = [self.wv_model[t] for t in tokens]
            vector = DocVecByWESum.compute_doc_vec(vectors)
        return vector

    def wv_update(self, docs):
        phrased_docs = list(self.aaer_model.get_trigrams(docs))
        self.wv_model.train(phrased_docs, total_examples=self.wv_model.corpus_count, epochs=self.wv_model.iter)
