"""This file contains deep learning models to achieve similar functions as context_based_models.py"""
from t2t_models import text_encoding
from common import constants as const
from common import file_tools as ft
from common import utilities as utils
from text_cleaning import example_parsing as ex_parsing
import t2t_make_data_files

import os
import logging
import pickle
import numpy as np


class ContextModel:
    def __init__(self):
        self._docvec_dict = {}

    def infer_vector(self, doc):
        raise NotImplementedError

    def _make_docvec_dict(self, docs):
        raise NotImplementedError


class T2TContextModel(ContextModel):
    @staticmethod
    def make_sure_docs_are_strings(docs):
        doc_type = type(docs[0])
        if doc_type is list or doc_type is tuple:
            str_docs = [' '.join(doc) for doc in docs]

        elif doc_type is str:
            str_docs = docs
        else:
            raise TypeError('doc type should either be list or str or tuple')
        return str_docs

    def __init__(self, load_aaer_test_data=False, doc_length=10, one_to_n=False, docs=None):
        """
        Args:
            load_aaer_test_data: if true, the model will automatically build/save/load docvec_dict with aaer data
            doc_length: used for Ngrams from aaer corpus. default is 10
            docs: not needed if load_aaer_data is True
        """
        super().__init__()
        self.load_aaer_test_data = load_aaer_test_data
        if load_aaer_test_data:
            self._load_aaer_test_data(doc_length, one_to_n=one_to_n)
        else:
            if docs:
                self.update(docs)

    def preferred_doc_length(self):
        key, _ = next(iter(self._docvec_dict))
        return len(key.split(' '))

    def infer_vector(self, doc):
        if type(doc) is list or type(doc) is tuple:
            doc = ' '.join(doc)
        try:
            vec = self._docvec_dict[doc]
        except KeyError:
            logging.info('not found in local data. using t2t model to infer...')
            self.update([doc])
            vec = self._docvec_dict[doc]
        return vec

    def infer_vectors_dict(self, docs):
        str_docs = self.make_sure_docs_are_strings(docs)
        # print(str_docs)
        self._make_docvec_dict(str_docs)
        vd = {utils.spaced_string_to_tuple(doc): self._docvec_dict[doc] for doc in str_docs}
        return vd

    def _make_docvec_dict(self, docs):
        str_docs = self.make_sure_docs_are_strings(docs)
        docs_to_update = []
        for str_doc in str_docs:
            if str_doc not in self._docvec_dict.keys():
                docs_to_update.append(str_doc)

        # str_docs[:] = [doc for doc in str_docs and doc not in self._docvec_dict.keys()]
        set_docs = set(docs_to_update)
        print('docs to update:')
        print(set_docs)
        if len(set_docs) > 0:
            docs_vocab = list(set_docs)
            encoding_model = text_encoding.TextEncoding(docs_vocab)
            encodings = encoding_model.encode(encoding_len=1)

            self._docvec_dict.update(dict(zip(docs_vocab, encodings)))
            logging.info("saving dict to %s" % self.dict_save_fname)
            with open(self.dict_save_fname, 'wb') as f:
                pickle.dump(self._docvec_dict, f)

    def update(self, docs):
        self._make_docvec_dict(docs=docs)

    def _load_aaer_test_data(self, doc_length, one_to_n=False):
        # data only contains test files, to save computing & memory costs
        self.save_dir = const.GENERATED_DATA_DIR
        if one_to_n:
            self.dict_save_fname = os.path.join(self.save_dir, "%s%s_1_to_%d.%s" %
                                                (const.DL_DOC_DICT_PREFIX, self.__class__.__name__,
                                                 doc_length, const.PICKLE_FILE_EXTENSION))
        else:
            self.dict_save_fname = os.path.join(self.save_dir, "%s%s_%d.%s" %
                                                (const.DL_DOC_DICT_PREFIX, self.__class__.__name__,
                                                 doc_length, const.PICKLE_FILE_EXTENSION))
        try:
            logging.info("loading saved data from %s" % self.dict_save_fname)
            with open(self.dict_save_fname, 'rb') as f:
                self._docvec_dict = pickle.load(f)
        except FileNotFoundError:
            logging.info("%s not found. Start building..." % self.dict_save_fname)
            test_files = ft.list_file_paths_under_dir(const.TEST_DIR, ['txt'])
            docs = []
            for test_file in test_files:
                if one_to_n:
                    docs += utils.flatten_list(
                        ex_parsing.one_to_n_grams_from_file(ft.get_source_file_by_example_file(test_file),
                                                            n=doc_length))
                else:
                    docs += ex_parsing.ngrams_from_file(ft.get_source_file_by_example_file(test_file), n=doc_length)
            # print(docs[0])
            self._make_docvec_dict(docs)


class ContextSimilarityT2TModel:
    def __init__(self, window_size=None):
        if window_size:
            self.window_size = window_size
        else:
            conf_dict = t2t_make_data_files.load_configs()
            self.window_size = conf_dict["window_size"]

    # find a file's ngrams' similarities, compared with given docs.
    # returns a dict whose keys are ngram_tuple from file, values are similarities given docs
    # (retains only highest one for unique ngram)
    def file_ngrams_similarities_by_docs(self, file_path, docs):
        ngram_dict = {}
        doc_similarity_dict = {}
        origin_sources = []
        origin_targets = []
        replaced_sources = []
        replaced_targets = []
        file_path = ft.get_source_file_by_example_file(file_path)
        for doc in docs:
            source_gram_n = len(doc)
            target_gram_n = t2t_make_data_files.get_target_gram_n(source_gram_n, self.window_size)
            try:
                target_ngrams = ngram_dict[target_gram_n]
            except KeyError:
                ngram_dict[target_gram_n] = target_ngrams = ex_parsing.ngrams_from_file(file_path, target_gram_n)
            source_ngrams = [t2t_make_data_files.source_ngram_from_target_ngram(target_ngram, self.window_size)
                             for target_ngram in target_ngrams]

            assert len(source_ngrams) == len(target_ngrams)
            origin_sources += source_ngrams
            origin_targets += target_ngrams
            for target in target_ngrams:
                replaced_target = t2t_make_data_files.replace_by_window_size(target, doc, self.window_size)
                replaced_targets.append(replaced_target)
                replaced_sources.append(doc)
        print("len(replaced_sources):%d" % len(replaced_sources))
        assert len(replaced_sources) == len(origin_sources) == len(replaced_targets) == len(origin_targets)
        # feed data into t2t model
        str_sources = [" ".join(tokens) for tokens in origin_sources + replaced_sources]
        str_targets = [" ".join(tokens) for tokens in origin_targets + replaced_targets]
        loss_model = text_encoding.TextSimilarity(str_sources, str_targets)
        losses = loss_model.encode()
        assert len(losses) == 2*len(origin_sources)
        origin_losses = np.array(losses[:len(origin_sources)])
        replaced_losses = np.array(losses[len(origin_sources):])
        print(origin_losses)
        print(replaced_losses)

