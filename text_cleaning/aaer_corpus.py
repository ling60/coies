import common.file_tools as ft
import common.constants as const
import pickle
import logging
import model_testing.word2vec_models as word2vec
import text_cleaning.example_parsing as ex_parsing
import gensim
import os

pickle_extension = '.' + const.PICKLE_FILE_EXTENSION


class AAERParserBase:
    def __init__(self):
        self.save_dir = const.GENERATED_DATA_DIR
        self.tokens_save_fname = os.path.join(self.save_dir, self.__class__.__name__ + pickle_extension)
        self.corpus_dir = os.path.join(const.DATA_PATH, const.AAER_PATH)
        self.word2vec_save_fname = os.path.join(self.save_dir, 'aaer_word2vec_' + self.get_word2vec_save_name())

    def get_word2vec_save_name(self):
        raise NotImplementedError

    def tokens_from_aaer_corpus(self):
        raise NotImplementedError

    def make_word2vec_model(self):
        assert self.word2vec_save_fname
        try:
            word2vec_model = gensim.models.Word2Vec.load(self.word2vec_save_fname)
        except FileNotFoundError:
            logging.info(self.word2vec_save_fname + ' not found')
            word2vec_model = word2vec.word2vec(self.get_tokens())
            word2vec_model.save(self.word2vec_save_fname)
        return word2vec_model

    def path_list_from_dir(self):
        return ft.list_file_paths_under_dir(self.corpus_dir, const.TEXT_EXTENSIONS)

    def get_tokens(self):
        try:
            logging.info('loading tokens from ' + self.tokens_save_fname)
            with open(self.tokens_save_fname, 'rb') as f:
                tokens = pickle.load(f)
        except FileNotFoundError:
            logging.info(self.tokens_save_fname + ' not found. Generating from data files...')
            tokens = self.tokens_from_aaer_corpus()
            with open(self.tokens_save_fname, 'wb') as f:
                pickle.dump(tokens, f)
        return tokens


class AAERParserNP(AAERParserBase):
    def get_word2vec_save_name(self):
        return 'np'

    def tokens_from_aaer_corpus(self):
        sentences = word2vec.sentences_from_file_list(ft.list_file_paths_under_dir(self.corpus_dir, ['txt']))
        return sentences


class AAERParserTokens(AAERParserBase):
    def get_word2vec_save_name(self):
        return 'tokens'

    def tokens_from_aaer_corpus(self):
        tokens = ex_parsing.tokens_from_dir(self.corpus_dir)
        return tokens


class AAERParserSentences(AAERParserBase):
    def get_word2vec_save_name(self):
        return 'sentences'

    def tokens_from_aaer_corpus(self):
        sentences = ex_parsing.sentences_from_dir(self.corpus_dir)
        return sentences


class AAERParserNGrams(AAERParserBase):
    def __init__(self, n=5):
        self.n = n
        super().__init__()
        self.tokens_save_fname = self.save_dir + self.__class__.__name__ + '_' + str(n) + pickle_extension

    def get_word2vec_save_name(self):
        return str(self.n) + 'grams'

    def tokens_from_aaer_corpus(self):
        ngrams = []
        for path in self.path_list_from_dir():
            ngrams += ex_parsing.ngrams_from_file(path, self.n)
        return ngrams


class AAERParserNGramsSkip(AAERParserNGrams):
    def __init__(self, n, n_skip):
        super().__init__(n)
        self.n_skip = n_skip
        self.tokens_save_fname = self.save_dir + self.__class__.__name__ + '_' + str(n) + '_' + str(
            n_skip) + pickle_extension

    def tokens_from_aaer_corpus(self):
        ngrams = []
        for path in self.path_list_from_dir():
            ng = ex_parsing.ngrams_from_file(path, self.n)
            for i in range(0, len(ng), self.n_skip):
                ngrams += ng[i]
        return ngrams


class AAERParserSequencedNGrams1ToN(AAERParserNGrams):
    def get_word2vec_save_name(self):
        return '1_to_' + str(self.n) + 'grams'

    def tokens_from_aaer_corpus(self):
        grams = []
        for path in self.path_list_from_dir():
            # for i in range(1, self.n + 1):
            #     grams.append([util.iter_to_string(tu) for tu in ex_parsing.sequenced_ngrams_from_file(path, i)])
            grams += ex_parsing.one_to_n_grams_from_file(path, self.n)
        return grams
