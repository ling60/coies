import common.file_tools as ft
import common.constants as const
import pickle
import logging
import model_testing.word2vec_models as word2vec
import text_cleaning.example_parsing as ex_parsing
import gensim
from gensim.models import phrases as gen_phrases
import os

pickle_extension = '.' + const.PICKLE_FILE_EXTENSION
CORPUS_EXTRA_DIR = os.path.join(const.DATA_PATH, const.EX_AAER_PATH)


class AAERParserBase:
    def __init__(self, corpus_dir=None):
        self.save_dir = const.GENERATED_DATA_DIR
        self.identify_str = None
        self.corpus_dir = corpus_dir if corpus_dir else os.path.join(const.DATA_PATH, const.AAER_PATH)
        self.word2vec_save_fname = os.path.join(self.save_dir, 'aaer_word2vec_' + self.get_word2vec_save_name())

    def get_tokens_save_path(self):
        file_name = self.__class__.__name__
        if self.identify_str:
            file_name = file_name + self.identify_str
        file_name = file_name + pickle_extension
        return os.path.join(self.save_dir, file_name)

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

    def get_tokens(self, enable_save=True):
        if enable_save:
            tokens_save_path = self.get_tokens_save_path()
            try:
                logging.info('loading tokens from ' + tokens_save_path)
                with open(tokens_save_path, 'rb') as f:
                    tokens = pickle.load(f)
            except FileNotFoundError:
                logging.info(tokens_save_path + ' not found. Generating from data files...')
                tokens = self.tokens_from_aaer_corpus()
                with open(tokens_save_path, 'wb') as f:
                    pickle.dump(tokens, f)
        else:
            logging.info("enable_save is False, generating data from files...")
            tokens = self.tokens_from_aaer_corpus()
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


class AAERExParserTokens(AAERParserTokens):
    def __init__(self):
        super().__init__(corpus_dir=CORPUS_EXTRA_DIR)


class AAERParserSentences(AAERParserBase):
    def get_word2vec_save_name(self):
        return 'sentences'

    def tokens_from_aaer_corpus(self):
        sentences = ex_parsing.sentences_from_dir(self.corpus_dir)
        return sentences


class AAERExParserSentences(AAERParserSentences):
    def __init__(self):
        super().__init__(corpus_dir=CORPUS_EXTRA_DIR)


# this class converts sentences to trigram phrases by gensim, added with two more methods:
# get_bigrams and get_trigrams, which return Phrases models to pick out phrases
class AAERParserPhrases(AAERParserBase):
    def __init__(self, corpus_dir=None):
        super().__init__(corpus_dir=corpus_dir)
        self.sentences = self.get_sentences()
        model_save_path = os.path.join(self.save_dir, "%s_model.pkl" % self.__class__.__name__)
        try:
            with open(model_save_path, 'rb') as f:
                self.bigrams, self.trigrams = pickle.load(f)
        except FileNotFoundError:
            self.bigrams = gen_phrases.Phrases(self.sentences)
            self.trigrams = gen_phrases.Phrases(self.bigrams[self.sentences])
            with open(model_save_path, 'wb') as f:
                pickle.dump((self.bigrams, self.trigrams), f)

    def get_word2vec_save_name(self):
        return 'phrases'

    def tokens_from_aaer_corpus(self):
        return list(self.trigrams[self.bigrams[self.sentences]])

    @staticmethod
    def get_sentences():
        return AAERParserSentences().get_tokens()

    def get_bigrams(self, sentences):
        assert type(sentences[0]) is list
        return self.bigrams[sentences]

    def get_trigrams(self, sentences):
        return self.trigrams[self.get_bigrams(sentences)]


class AAERExParserPhrases(AAERParserPhrases):
    def __init__(self):
        super().__init__(corpus_dir=CORPUS_EXTRA_DIR)

    @staticmethod
    def get_sentences():
        return AAERExParserSentences().get_tokens()


class AAERParserNGrams(AAERParserBase):
    def __init__(self, n=5, corpus_dir=None):
        self.n = n
        super().__init__(corpus_dir=corpus_dir)
        self.identify_str = "_%d" % n

    def get_word2vec_save_name(self):
        return str(self.n) + 'grams'

    def tokens_from_aaer_corpus(self):
        ngrams = []
        for path in self.path_list_from_dir():
            ngrams += ex_parsing.ngrams_from_file(path, self.n)
        return ngrams


class AAERExParserNGrams(AAERParserNGrams):
    def __init__(self, n=5, corpus_dir=CORPUS_EXTRA_DIR):
        super().__init__(n=n, corpus_dir=corpus_dir)
        self.identify_str = "_ex_%d" % n


class AAERParserM2NGrams(AAERParserBase):
    def __init__(self, m=1, n=5, corpus_dir=None):
        self.n = n
        self.m = m
        super().__init__(corpus_dir=corpus_dir)
        self.identify_str = "_%d_%d" % (m, n)

    def get_word2vec_save_name(self):
        return "%d_%d_grams" % (self.m, self.n)

    def tokens_from_aaer_corpus(self):
        ngrams = []
        for path in self.path_list_from_dir():
            ngrams += ex_parsing.m_to_n_grams_from_file(path, m=self.m, n=self.n)
        return ngrams


class AAERExParserM2NGrams(AAERParserM2NGrams):
    def __init__(self, m=1, n=5, corpus_dir=CORPUS_EXTRA_DIR):
        super().__init__(m=m, n=n, corpus_dir=corpus_dir)
        self.identify_str = "_ex%s" % self.identify_str


class AAERParserNGramsSkip(AAERParserNGrams):
    def __init__(self, n, n_skip):
        super().__init__(n)
        self.n_skip = n_skip
        self.identify_str = "_%d_skip_%d" % (n, n_skip)

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
