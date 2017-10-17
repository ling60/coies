"""This file contains deep learning models to achieve similar functions as context_based_models.py"""
from t2t_models import text_encoding
from text_cleaning import aaer_corpus
from common import constants as const

import os
import logging
import pickle


class ContextModel:
    def __init__(self):
        self._docvec_dict = {}

    def infer_vector(self, doc):
        raise NotImplementedError

    def _make_docvec_dict(self, docs):
        raise NotImplementedError


class T2TContextModel(ContextModel):
    def __init__(self, load_aaer_data=False, doc_length=10, docs=None):
        """
        Args:
            load_aaer_data: if true, the model will automatically build/save/load docvec_dict with aaer data
            doc_length: used for Ngrams from aaer corpus. default is 10
            docs: not needed if load_aaer_data is True
        """
        super().__init__()
        if load_aaer_data:
            self.save_dir = const.GENERATED_DATA_DIR
            self.dict_save_fname = os.path.join(self.save_dir, "dl_doc_dict_%s_%d.%s" %
                                                  (self.__class__.__name__, doc_length, const.PICKLE_FILE_EXTENSION))
            try:
                logging.info("loading saved data from %s" % self.dict_save_fname)
                with open(self.dict_save_fname, 'rb') as f:
                    self._docvec_dict = pickle.load(f)
            except FileNotFoundError:
                logging.info("%s not found. Start building...")
                aaer = aaer_corpus.AAERParserNGrams(n=doc_length)
                docs = aaer.get_tokens()
                self._make_docvec_dict(docs)
                logging.info("saving dict to %s" % self.dict_save_fname)
                with open(self.dict_save_fname, 'wb') as f:
                    pickle.dump(self._docvec_dict, f)

    def preferred_doc_length(self):
        key, _ = next(iter(self._docvec_dict))
        return len(key.split(' '))

    def infer_vector(self, doc):
        pass

    def _make_docvec_dict(self, docs):
        doc_type = type(docs[0])
        if doc_type is list:
            str_docs = [' '.join(doc) for doc in docs]
        elif doc_type is str:
            str_docs = docs
        else:
            raise TypeError('doc type should either be list or str')
        docs_vocab = list(set(str_docs))
        encoding_model = text_encoding.TextEncoding(docs_vocab)
        encodings = encoding_model.encode()

        self._docvec_dict = dict(zip(str_docs, encodings))
