import gensim
import common.constants as const
import common.file_tools as ft
import common.utilities as util
import model_testing.word2vec_models as word2vec
import model_testing.context_based_models as cb
import text_cleaning.example_parsing as ex_parsing
import root_path
import text_cleaning.aaer_corpus as aaer
import logging
import os
import model_testing.rougescore as rouge


def get_current_path():
    return os.path.dirname(os.path.abspath(__file__))


def score_by_hits(words_found, test_entity_dict, entity_key):
    logging.info('score_by_hits: words_found: ' + str(words_found))
    hits = 0
    targets = 0
    if entity_key in test_entity_dict:
        for word in words_found:
            targets += len(test_entity_dict[entity_key])
            if word in sum(test_entity_dict[entity_key], []):
                hits += 1
    return hits, targets


def score_by_rouge(words_found, test_entity_dict, entity_key):
    logging.info('score_by_hits: words_found: ' + str(words_found))
    score = 0
    targets = 0
    if entity_key in test_entity_dict:
        targets += len(test_entity_dict[entity_key])

        score += rouge.rouge_1(words_found, test_entity_dict[entity_key], alpha=0.5)

    return score, targets


class OneShotTestBase:
    def __init__(self, example_path, test_file_path_list, enable_saving=False, n_gram=5):
        util.display_logging_info()
        self.score_dict = {}
        self.example_path = example_path
        self.example_tokens, self.example_entity_dict = self.tokens_entities_from_path(example_path)
        self.test_file_path_list = test_file_path_list
        self.example_wv = None
        self.enable_saving = enable_saving
        self.n = n_gram
        self.doc2vec_model = None

    @staticmethod
    # def tokens_entities_from_path(file_path):
    #     tokens, entity_dict = ex_parsing_np.parse_file(file_path)
    #     entity_dict = word2vec.remove_punctuations_from_entity_dict(entity_dict)
    #     return tokens, entity_dict
    def tokens_entities_from_path(file_path):
        tokens = ex_parsing.one_to_n_grams_from_file(file_path, tagged=True)
        # print(tokens[1][0])
        entity_dict = ex_parsing.entity_dict_from_tagged_tokens(ex_parsing.tagged_tokens_from_file(file_path))
        # logging.info(tokens)
        return tokens, entity_dict

    def make_example_word_vectors(self):
        raise NotImplementedError

    def make_test_wv_dict(self, test_tokens):
        raise NotImplementedError

    def similar_words_by_word(self, word, wv_dict, topn=3):
        # logging.info('similiar_words_by_word: word: ' + word + 'text_wv_dict: ' + str(test_wv_dict))
        words_found = [w[0] for w in util.similar_by_vector(self.example_wv[word], wv_dict, topn=topn)]
        return words_found

    def test(self):
        assert self.example_entity_dict
        # do the test
        self.example_wv = self.make_example_word_vectors()
        # print(example_entity_dict)
        total_targets = 0
        for test_file_path in self.test_file_path_list:
            logging.info('testing file:' + test_file_path)
            self.score_dict[test_file_path] = 0
            test_tokens, test_entity_dict = self.tokens_entities_from_path(test_file_path)
            logging.info('test_entity_dict')
            logging.info(test_entity_dict)
            test_wv_dict = self.make_test_wv_dict(test_tokens)

            for k, value_lists in self.example_entity_dict.items():
                for values in value_lists:
                    if type(values) is list:
                        for v in values:
                            # find most similar words in test file
                            print('similar to:' + v)
                            words_found = self.similar_words_by_word(v, test_wv_dict)
                            hits, targets = score_by_rouge(words_found, test_entity_dict, k)
                            print("rouge:", hits)
                            self.score_dict[test_file_path] += hits
                            total_targets += targets
                    else:
                        raise Exception('the value of entity dict should be list')
        print(self.score_dict)
        total_score = sum(self.score_dict.values())
        print('total score', sum(self.score_dict.values()), 'of', total_targets, 'in',
              len(self.score_dict.keys()), 'files')
        return total_score, total_targets


# example vectors: the example file; test vectors: the test file
class OneShotTest1(OneShotTestBase):
    def make_example_word_vectors(self):
        return word2vec.word2vec(self.example_tokens)

    def make_test_wv_dict(self, test_tokens):
        test_model = word2vec.word2vec(test_tokens)
        return util.word_vector_to_dict_by_list(test_model.wv, util.flatten_list(test_tokens))


# example & test vectors: all test files + example file
class OneShotTest2(OneShotTestBase):
    def make_example_word_vectors(self):
        return word2vec.word2vec(word2vec.sentences_from_file_list(self.test_file_path_list))

    def make_test_wv_dict(self, test_tokens):
        #  return util.word_vector_to_dict_by_list(self.example_wv, util.flatten_list(test_tokens))
        # print(test_tokens)
        with open('temp.txt', 'w') as f:
            f.write(str(test_tokens))
        return util.word_vector_to_dict_by_list(self.example_wv, util.flatten_list(test_tokens))


# example & test vectors: all aaer files
# baseline: 351 of 1818 in 100 files
# rouge1 score: 104
class OneShotTest3(OneShotTestBase):
    def __init__(self, example_path, test_file_path_list, enable_saving):
        super().__init__(example_path, test_file_path_list, enable_saving)
        self.save_dir = ft.check_dir_ending(root_path.GENERATED_DATA_DIR)
        self.save_fname = self.save_dir + self.__class__.__name__ + '.sav'

    def make_doc2vec_model(self):
        save_fname = ft.check_dir_ending(root_path.GENERATED_DATA_DIR) + 'aaer_doc2vec_' + str(self.n) + 'grams'
        try:
            doc_vec_model = gensim.models.Doc2Vec.load(save_fname)
        except FileNotFoundError:
            logging.info(save_fname + ' not found')
            labeled_tagged_ngrams = cb.label_ngrams_from_file_list(ft.list_file_paths_under_dir(
                const.DATA_PATH + const.AAER_PATH, ['txt']))
            doc_vec_model = cb.doc2vec(labeled_tagged_ngrams)
            doc_vec_model.save(save_fname)
        return doc_vec_model

    def make_example_word_vectors(self):
        return word2vec.word2vec(word2vec.sentences_from_file_list(self.test_file_path_list))

    def make_test_wv_dict(self, test_tokens):
        #  return util.word_vector_to_dict_by_list(self.example_wv, util.flatten_list(test_tokens))
        # print(test_tokens)
        return util.word_vector_to_dict_by_list(self.example_wv, util.flatten_list(test_tokens))

    def get_tokens(self):
        # aaer_parser = aaer.AAERParserNP()
        aaer_parser = aaer.AAERParserSequencedNGrams1ToN()
        return aaer_parser.get_tokens()

    @staticmethod
    def word_vec_from_tokens(tokens):
        wv = word2vec.word2vec(tokens)
        return wv

    def make_word_vectors_and_save(self):
        tokens = self.get_tokens()
        wv = self.word_vec_from_tokens(tokens)
        if self.enable_saving:
            wv.save(self.save_fname)
        return wv

    def make_example_word_vectors(self):
        if self.enable_saving:
            try:
                wv = gensim.models.KeyedVectors.load(self.save_fname)
            except FileNotFoundError:
                logging.info(self.save_fname + ' not found')
                wv = self.make_word_vectors_and_save()
        else:
            wv = self.make_word_vectors_and_save()
        return wv


# test different configs of word vector training
# sg=1(skip model) perform no better than cbow with small corpus
# increasing dimensions from 100 to 300 yields a little better result (10 more hits), iter from 5 - 10 gets 5 more
# increasing window size from 5 - 10 does not change much, so does negative sampling
class OneShotTest3a(OneShotTest3):
    @staticmethod
    def word_vec_from_tokens(tokens):
        return word2vec.word2vec(tokens, size=300)


# test what happens of giving more weights to example & test files
# simply duplicating example files shows no improvement
class OneShotTest3b(OneShotTest3):
    def get_tokens(self):
        weighted_tokens = super().get_tokens()
        weighted_tokens += self.example_tokens * 4
        logging.info(weighted_tokens[-5:])
        return weighted_tokens


# test what happens of giving more weights to example & test files
# make extra training with given files
class OneShotTest3c(OneShotTest3):
    def make_word_vectors_and_save(self):
        model_save_file = 'model_' + self.save_fname
        if not os.path.exists(model_save_file):
            tokens = self.get_tokens()
            model = word2vec.word2vec(tokens)
            model.save(model_save_file)
        else:
            model = word2vec.word2vec(None)
            model.load(model_save_file)

        model.train(self.example_tokens*4)
        wv = model.wv
        del model
        if self.enable_saving:
            wv.save(self.save_fname)
        return wv


# using pre-trained wiki model, then train with aaer corpus
class OneShotTest4(OneShotTest3):
    def __init__(self, example_path, test_file_path_list, enable_saving):
        super().__init__(example_path, test_file_path_list, enable_saving)
        self.wiki_aaer_vec_path = self.save_dir + 'fasttext_wiki_aaer.vec'

    def make_word_vectors_and_save(self):
        tokens = self.get_tokens()
        if not os.path.exists(self.wiki_aaer_vec_path):
            logging.info(self.wiki_aaer_vec_path + ' does not exist. Trying to build one...')
            word2vec.make_vec_file_from_wiki_model(tokens, self.wiki_aaer_vec_path)

        model = gensim.models.Word2Vec.load_word2vec_format(self.wiki_aaer_vec_path)
        model.train(tokens, min_count=1)
        wv = model.wv
        if self.enable_saving:
            wv.save(self.save_fname)
        return wv


# example & test vectors: fast text vectors from all aaer files
# the test is worse:278 hits
class OneShotTest5(OneShotTest2):
    def make_example_word_vectors(self):
        wv = word2vec.fasttext_model_from_file2('../text_cleaning/aaer_.txt')
        return wv


# using our own parser without nltk. To parse text into sentences(2d-list) improves the result from 98 to 348, same
# level as nltk sentences parser.
class OneShotTest10(OneShotTest3):
    @staticmethod
    def tokens_entities_from_path(file_path):
        tokens = ex_parsing.tokens_from_file(file_path)
        entity_dict = ex_parsing.entity_dict_from_tagged_tokens(ex_parsing.tagged_tokens_from_file(file_path))
        # logging.info(tokens)
        return tokens, entity_dict

    @staticmethod
    def word_vec_from_tokens(tokens):
        wv = word2vec.word2vec(tokens)
        return wv

    def make_test_wv_dict(self, test_tokens):
        return util.word_vector_to_dict_by_list(self.example_wv, test_tokens)

    def get_tokens(self):
        aaer_parser = aaer.AAERParserSentences()
        logging.info(aaer_parser.get_tokens())
        return aaer_parser.get_tokens()


# ngrams = 10 yields same level of result as sentences. ngrams = 5 shows no difference
class OneShotTest10a(OneShotTest10):
    def get_tokens(self):
        aaer_parser = aaer.AAERParserNGrams(n=5)
        return aaer_parser.get_tokens()


# computing doc2vec and then word2vec
# setting doc2vec size=100: 268
# setting doc2vec size=300: 276
# rouge1 score: 84
class OneShotTest11(OneShotTest10):
    def __init__(self, example_path, test_file_path_list, enable_saving=False):
        super().__init__(example_path, test_file_path_list, enable_saving)
        self.tagged_tokens = ex_parsing.tagged_tokens_from_file(self.example_path)
        self.example_entity_tagged_words_dict = \
            ex_parsing.entity_tagged_words_dict_from_tagged_tokens(self.tagged_tokens)
        self.example_ngrams = ex_parsing.ngrams_from_file(self.example_path, self.n, tagged=True)
        self.n = 10

    @staticmethod
    def tagged_words_to_str(tagged_words):
        return const.UNIQUE_DELIMITER.join(util.flatten_list(tagged_words))

    @staticmethod
    # returns a set of words, constructed from ngrams in dv dict similar to doc vecs
    def similar_words_by_doc_vecs(doc_vecs, dv_dict, topn=3):
        # logging.info('similiar_words_by_word: word: ' + word + 'text_wv_dict: ' + str(test_wv_dict))
        words_found = set()
        for doc_vec in doc_vecs:
            l = [w[0].split(const.UNIQUE_DELIMITER) for w in util.similar_by_vector(doc_vec, dv_dict, topn=topn)]
            words_found.update(set(util.flatten_list(l)))
        return words_found

    # make a dict which is convenient to look up a list of ngram vectors by tagged_words
    def make_example_tagged_words_ngram_vecs_dict(self):
        assert self.example_entity_tagged_words_dict
        example_tagged_words_ngram_vecs_dict = {}
        for entity, value_lists in self.example_entity_tagged_words_dict.items():
            for tagged_words in value_lists:
                if type(tagged_words) is list:
                    str_tagged_words = self.tagged_words_to_str(tagged_words)
                    example_tagged_ngrams = cb.find_ngrams_by_tagged_words(self.example_ngrams, tagged_words)
                    example_tagged_words_ngram_vecs_dict[str_tagged_words] =\
                        [self.doc2vec_model.infer_vector(ngram) for ngram in example_tagged_ngrams]
        return example_tagged_words_ngram_vecs_dict

    def test(self):
        # a dict like {'comp': [[['esafetyworld', 'comp'], ['inc', 'end']],[...]]], 'date': [[['2000', 'date']],
        # [['2001', 'date']]], 'item': [[['revenues', 'item']], [['profits', 'item']]]}
        assert self.example_entity_tagged_words_dict
        # do the test
        self.example_wv = self.make_example_word_vectors()
        self.doc2vec_model = self.make_doc2vec_model()
        # print(example_entity_dict)
        total_targets = 0
        example_tagged_words_ngram_vecs_dict = self.make_example_tagged_words_ngram_vecs_dict()
        for test_file_path in self.test_file_path_list:
            self.score_dict[test_file_path] = 0
            test_tokens, test_entity_dict = self.tokens_entities_from_path(test_file_path)
            logging.info('test_entity_dict')
            logging.info(test_entity_dict)
            test_wv_dict = self.make_test_wv_dict(test_tokens)

            test_file_ngrams_with_tags = ex_parsing.ngrams_from_file(test_file_path, self.n, tagged=True)
            test_file_ngrams_without_tags = [util.sentence_from_tagged_ngram(t) for t in test_file_ngrams_with_tags]
            test_ngram_vec_dict = cb.doc_vector_dict_by_ngrams(self.doc2vec_model, test_file_ngrams_without_tags)

            for entity, value_lists in self.example_entity_tagged_words_dict.items():
                for tagged_words in value_lists:
                    assert type(tagged_words) is list
                    # find ngrams in test file similar to example
                    example_tagged_words_ngram_vecs = \
                        example_tagged_words_ngram_vecs_dict[self.tagged_words_to_str(tagged_words)]
                    context_word_set = self.similar_words_by_doc_vecs(
                        example_tagged_words_ngram_vecs, test_ngram_vec_dict)

                    context_wv_dict = util.subset_dict_by_list(test_wv_dict, context_word_set)
                    logging.info('context_wv_dict:')
                    logging.info(len(context_wv_dict))
                    for v in tagged_words:
                        # find most similar words in test file
                        v = v[0]
                        print('similar to:' + v)
                        words_found = self.similar_words_by_word(v, context_wv_dict)
                        hits, targets = score_by_rouge(words_found, test_entity_dict, entity)
                        print("rouge:", hits)
                        self.score_dict[test_file_path] += hits
                        total_targets += targets

        logging.info('score_dict')
        logging.info(self.score_dict)
        total_score = sum(self.score_dict.values())
        print('total score', sum(self.score_dict.values()), 'of', total_targets, 'in',
              len(self.score_dict.keys()), 'files')
        return total_score, total_targets
