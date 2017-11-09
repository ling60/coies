import text_cleaning.aaer_corpus as aaer
import common.constants as const
import common.utilities as util
import common.file_tools as ft
import random
import model_testing.context_based_models as cb
import text_cleaning.example_parsing as ex_parsing
import logging
import os
import model_testing.rougescore as rouge
import model_testing.dl_context_models as dl_context


base_conf_dict = {
    'topn': 3,
    'context_threshold': 0.75,
    'word_threshold': 0.4,
    'context_size': 20
}


def get_current_path():
    return os.path.dirname(os.path.abspath(__file__))


def similar_grams_by_vec(vec, wv_dict, topn, similarity_threshold):
    # logging.info('similar_words_by_word: word: ' + word + 'text_wv_dict: ' + str(test_wv_dict))
    gram_value_tuples = util.similar_by_vector(vec, wv_dict, topn=topn)
    # print(gram_value_tuples)
    # for grams, similarity in util.similar_by_vector(vec, wv_dict, topn=topn):
    #     # logging.info(similarity)
    #     if similarity > similarity_threshold:
    #         gram_value_tuples.append(grams)
    gram_value_tuples = util.get_top_group(gram_value_tuples, similarity_threshold)
    # print(gram_value_tuples)
    # return [t[0] for t in gram_value_tuples]
    # print(gram_value_tuples)
    return gram_value_tuples


# returns a list of  ngrams in dv dict similar to doc vecs
def similar_grams_by_doc_vecs(doc_vecs, dv_dict, topn, sim_threshold):
    # logging.info('similiar_words_by_word: word: ' + word + 'text_wv_dict: ' + str(test_wv_dict))
    grams_similarity_dict = {}
    for doc_vec in doc_vecs:
        gram_value_tuples = similar_grams_by_vec(doc_vec, dv_dict, topn=topn, similarity_threshold=sim_threshold)
        for gram, similarity in gram_value_tuples:
            if type(gram) is str:
                gram = gram.split(' ')
            gram_tuple = tuple(gram)
            try:
                similarity_existed = grams_similarity_dict[gram_tuple]
                if similarity > similarity_existed:
                    grams_similarity_dict[gram_tuple] = similarity
            except KeyError:
                grams_similarity_dict[gram_tuple] = similarity
    return grams_similarity_dict


# returns two dicts based on context similar to given ngram vectors. first is word vector dict filtered by contexts,
# the other is contexts and their similarity values
def make_context_sim_dict(example_tagged_words_ngram_vecs, context_sized_test_wv_dict, topn, sim_threshold):
    return similar_grams_by_doc_vecs(example_tagged_words_ngram_vecs, 
                                     context_sized_test_wv_dict,
                                     topn=topn,
                                     sim_threshold=sim_threshold)


def make_context_dict(example_tagged_words_ngram_vecs, context_sized_test_wv_dict, wv_dict, topn, sim_threshold):
    context_similarity_dict = make_context_sim_dict(example_tagged_words_ngram_vecs, 
                                                    context_sized_test_wv_dict, 
                                                    topn,
                                                    sim_threshold)
    logging.info('similar contexts:')
    logging.info(context_similarity_dict)
    # similar_contexts = set()
    context_wv_dict = util.subset_dict_by_list2(wv_dict, context_similarity_dict.keys())
    # print(context_wv_dict.keys())
    return context_wv_dict, context_similarity_dict


def get_entity_dict_from_file(file_path):
    return ex_parsing.entity_dict_from_tagged_tokens(ex_parsing.tagged_tokens_from_file(file_path))


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


# add score for no words found when there is no answer in example either
def score_by_rouge(words_found, test_entity_dict, entity_key):
    logging.info('score_by_Rouge: words_found: ' + str(words_found))
    score = (0, 0)
    targets = 1
    if entity_key in test_entity_dict:
        answers = [util.flatten_list(test_entity_dict[entity_key])]
        words_found = [] if words_found is None else util.flatten_list(words_found)
        # targets += len(test_entity_dict[entity_key])
        print('answers:')
        print(answers)
        # print(words_found)
        score = util.tuple_add(score, (rouge.rouge_1(words_found, answers, alpha=0.5),
                                       rouge.rouge_2(words_found, answers, alpha=0.5)))
        # print(score)
    elif not words_found:  # both do not have similar words compared to example
        score = util.tuple_add(score, (1, 0))  # set rouge2 as 0 because for single word rouge2 returns 0
    return score, targets


# Basic setting, with a doc-vec trained on 5 grams AAER corpus. Rouge_1 test score is 7.4
# total score (7.652952038250112, 2.073774873774874) of 531 in 100 files
class OneShotTestDoc2Vec:
    def __init__(self, example_path, test_file_path_list, enable_saving=False, n_gram=5, **kwargs):
        # util.display_logging_info()
        self.score_dict = {}
        self.example_path = example_path
        self.example_entity_dict = get_entity_dict_from_file(example_path)
        self.test_file_path_list = test_file_path_list
        self.enable_saving = enable_saving
        self.n = n_gram
        self.doc_vec_model = None
        self.test_tokens = None
        self.test_entity_dict = None
        self.test_wv_dict = None

        if "enable_phrases" in kwargs:
            if kwargs["enable_phrases"]:
                pass
        else:
            # init phrases detector
            self.phrases_model = aaer.AAERExParserPhrasesBigrams()
        if "conf_dict" in kwargs:
            self.conf_dict = kwargs['conf_dict']
            self.topn = self.conf_dict['topn']
            self.word_threshold = self.conf_dict['word_threshold']
            self.context_threshold = self.conf_dict['context_threshold']
            self.context_size = self.conf_dict['context_size']

    def tokens_entities_from_path(self, file_path):
        tokens = ex_parsing.one_to_n_grams_from_file(file_path, n=self.n, tagged=True)
        # print(tokens[1])
        entity_dict = get_entity_dict_from_file(file_path)
        # logging.info(tokens)
        return tokens, entity_dict

    @staticmethod
    def doc_vector_to_dict_by_list(dv_model, grams):
        return {tuple(k): dv_model.infer_vector(k) for k in grams}

    def doc_vectors_training(self):
        return cb.make_doc2vec_model_from_aaer()

    def make_test_wv_dict(self, test_grams):
        flat_grams = util.flatten_list(test_grams)
        flat_grams[:] = [tuple(w.split(const.GENSIM_PHRASES_DELIMITER)) for w in flat_grams]
        self.test_wv_dict = self.doc_vector_to_dict_by_list(self.doc_vec_model, flat_grams)

    def init_score_dict(self, test_file_path):
        self.score_dict[test_file_path] = (0, 0)

    def similar_grams_by_gram(self, gram, wv_dict, similarity_threshold=None):
        similarity_threshold = similarity_threshold if similarity_threshold else self.word_threshold
        return similar_grams_by_vec(self.doc_vec_model.infer_vector(gram),
                                    wv_dict,
                                    topn=self.topn,
                                    similarity_threshold=similarity_threshold)

    def score(self, key, gram, test_file_path, wv_dict, **kwargs):
        # todo: filter out unnatural terms, such as "profit as"
        print('similar to:' + str(gram))
        if 'context_sim_dict' in kwargs and False:  # disable weighted func
            # allocate weights to grams coming from different contexts
            context_sim_dict = kwargs['context_sim_dict']
            weighted_wv_dict = {}
            gram_vector = self.doc_vec_model.infer_vector(gram)
            distance_dict = util.make_distance_dict(gram_vector, wv_dict)
            for context, similarity in context_sim_dict.items():
                for w, distance, in distance_dict.items():
                    if util.is_sublist_of(w, context):
                        # new_weighted_distance = 1 - (1 - similarity) * (1 - distance)
                        new_weighted_distance = (1 + similarity) * distance
                        # we add a "1" here to ensure the result is larger
                        try:
                            weighted_distance = weighted_wv_dict[w]
                            if weighted_distance < new_weighted_distance:
                                weighted_wv_dict[w] = new_weighted_distance
                        except KeyError:
                            weighted_wv_dict[w] = new_weighted_distance
            # print(weighted_wv_dict)
            # print(wv_dict)
            word_similarity_tuples = util.get_top_group(util.most_common_items(weighted_wv_dict, 
                                                                               topn=self.topn),
                                                        distance_threshold=self.word_threshold*2)
            # the threshold is doubled as similarity increased after adding context weights
        else:
            word_similarity_tuples = self.similar_grams_by_gram(gram, wv_dict)
        print(word_similarity_tuples)
        # print(wv_dict.keys())

        words_found = [t[0] for t in word_similarity_tuples]
        print('words found:')
        print(words_found)
        print('correct:')
        try:
            print(self.test_entity_dict[key])
        except KeyError:
            print('None')
        # if len(words_found) == 0:
        #     words_found = []
        # else:
        #     words_found = [words_found[0]]
        scores, counts = score_by_rouge(words_found, self.test_entity_dict, key)
        print("rouge:", scores)
        self.score_dict[test_file_path] = util.tuple_add(self.score_dict[test_file_path], scores)
        return counts

    def train(self):
        assert self.example_entity_dict
        self.doc_vec_model = self.doc_vectors_training()

    def test_file_processing(self, test_file_path):
        logging.info('testing file:' + test_file_path)
        self.init_score_dict(test_file_path)
        sentences = ex_parsing.sentences_from_file(ft.get_source_file_by_example_file(test_file_path))
        # todo: change bigrams to trigrams, after test!
        self.test_tokens = self.phrases_model.get_bigrams(sentences)
        self.test_entity_dict = get_entity_dict_from_file(test_file_path)
        # self.test_tokens, self.test_entity_dict = self.tokens_entities_from_path(test_file_path)
        logging.info('test_entity_dict')
        logging.info(self.test_entity_dict)
        self.make_test_wv_dict(self.test_tokens)

    def test(self):
        # print(example_entity_dict)
        total_counts = 0
        for test_file_path in self.test_file_path_list:
            if test_file_path != self.example_path:
                self.test_file_processing(test_file_path)
                counts = 0
                for k, value_lists in self.example_entity_dict.items():
                    for gram in value_lists:
                        # find most similar words in test file
                        assert type(gram) is list
                        counts += self.score(k, gram, test_file_path, self.test_wv_dict)
                if counts > 0:
                    rouge1, rouge2 = self.score_dict[test_file_path]

                    self.score_dict[test_file_path] = (rouge1 / counts, rouge2 / counts)
                total_counts += counts

        print(self.score_dict)
        total_score = tuple(map(sum, zip(*self.score_dict.values())))
        print('total score', str(total_score), 'of', total_counts, 'in',
              len(self.score_dict.keys()), 'files')
        return total_score, total_counts


# try the perfect score:
# total score (294.13333333333344, 158.338547978175) of 531 in 100 files
# [ 73.54138232  23.87471053] with new score
# [ 100.    54.8]
# 99, 53.8
class OneShotTestPerfect(OneShotTestDoc2Vec):
    def __init__(self, example_path, test_file_path_list, enable_saving=False, n_gram=5, **kwargs):
        super().__init__(example_path, test_file_path_list, enable_saving, n_gram, enable_phrases=False, **kwargs)

    def make_test_wv_dict(self, test_grams):
        pass

    def train(self):
        pass

    def test_file_processing(self, test_file_path):
        logging.info('testing file:' + test_file_path)
        self.init_score_dict(test_file_path)
        self.test_entity_dict = get_entity_dict_from_file(test_file_path)
        # self.test_tokens, self.test_entity_dict = self.tokens_entities_from_path(test_file_path)
        logging.info('test_entity_dict')
        logging.info(self.test_entity_dict)

    def score(self, key, gram, test_file_path, wv_dict, **kwargs):
        print('similar to:' + str(gram))
        # words_found = self.similar_grams_by_gram(gram, gram)
        try:
            answers = self.test_entity_dict[key]
        except KeyError:
            answers = []
        score, counts = score_by_rouge(answers, self.test_entity_dict, key)
        print("rouge:", score)
        self.score_dict[test_file_path] = util.tuple_add(self.score_dict[test_file_path], score)
        return counts


# try the score based on random selection:

# [ 3.51328195  0.63288726] of 531 in 100 files (average of 100 epochs)
class OneShotTestRandom(OneShotTestPerfect):
    def score(self, key, gram, test_file_path, wv_dict, **kwargs):
        print('similar to:' + str(gram))
        # words_found = self.similar_grams_by_gram(gram, gram)
        answers = [random.choice(util.flatten_list(self.test_tokens))]
        hits, targets = score_by_rouge(answers, self.test_entity_dict, key)
        print("rouge:", hits)
        self.score_dict[test_file_path] = util.tuple_add(self.score_dict[test_file_path], hits)
        return targets


# test the score if just return empty results
# [ 29.82857143   0.        ] with new score
class OneShotTestNone(OneShotTestPerfect):
    def score(self, key, gram, test_file_path, wv_dict, **kwargs):
        print('similar to:' + str(gram))
        # words_found = self.similar_grams_by_gram(gram, gram)
        answers = []
        hits, targets = score_by_rouge(answers, self.test_entity_dict, key)
        print("rouge:", hits)
        self.score_dict[test_file_path] = util.tuple_add(self.score_dict[test_file_path], hits)
        return targets


# total score (214.93262997209598, 113.0204099518053) of 531 in 100 files
# [ 38.63431138  12.39807115] with new score
# [ 28.65692071  12.39268394] with rouge1 = 0 for None
# [ 47.55615901  25.20420298]
class OneShotTestHuman(OneShotTestPerfect):
    def score(self, key, gram, test_file_path, wv_dict, **kwargs):
        human_file_path = os.path.join(const.HUMAN_DIR, test_file_path.split('/')[-1])
        test_entity_dict = get_entity_dict_from_file(human_file_path)
        try:
            answers = test_entity_dict[key]
        except KeyError:
            answers = ""
        hits, targets = score_by_rouge(answers, self.test_entity_dict, key)
        print("rouge:", hits)
        self.score_dict[test_file_path] = util.tuple_add(self.score_dict[test_file_path], hits)
        return targets


# instead of comparing words directly, we will choose most similar context(sentences) first, then find similar words
# from given context.
# this model shares same doc vectors between context and words, which is trained in ngrams = context_size
# total score of 531 in 100 files
# 5.62889347  1.8
class OneShotTestContext1(OneShotTestDoc2Vec):
    def __init__(self, example_path, test_file_path_list, enable_saving=False, n_gram=5, **kwargs):
        super().__init__(example_path, test_file_path_list, enable_saving, n_gram, **kwargs)
        # self.context_size = context_size
        self.context_vec_model = None
        self.tagged_tokens = ex_parsing.tagged_tokens_from_file(self.example_path)
        self.example_entity_dict = \
            ex_parsing.entity_tagged_words_dict_from_tagged_tokens(self.tagged_tokens)
        self.example_ngrams = ex_parsing.ngrams_from_file(self.example_path, self.context_size, tagged=True)
        self.example_tagged_words_ngram_vecs_dict = {}
        self.context_sized_test_wv_dict = None

    def train(self):
        super().train()
        self.post_training()

    def context_doc_training(self):
        return self.doc_vec_model

    def post_training(self):
        self.context_vec_model = self.context_doc_training()
        self.make_example_tagged_words_ngram_vecs_dict(self.doc_vec_model)

    def doc_vectors_training(self):
        return cb.make_doc2vec_model_from_aaer()

    @staticmethod
    def tagged_words_to_str(tagged_words):
        return const.UNIQUE_DELIMITER.join(util.flatten_list(tagged_words))

    # @staticmethod
    # # returns a list of  ngrams in dv dict similar to doc vecs
    # def similar_grams_by_doc_vecs(doc_vecs, dv_dict, topn=TOP_N):
    #     # logging.info('similiar_words_by_word: word: ' + word + 'text_wv_dict: ' + str(test_wv_dict))
    #     grams_found = set()
    #     for doc_vec in doc_vecs:
    #         grams = similar_grams_by_vec(doc_vec, dv_dict, topn=topn)
    #         grams_found.update(set(tuple(gram) for gram in grams))
    #     return grams_found

    # make a dict which is convenient to look up a list of ngram vectors by tagged_words
    def make_example_tagged_words_ngram_vecs_dict(self, doc2vec_model):
        assert self.example_entity_dict
        example_tagged_words_ngram_vecs_dict = {}
        for entity, value_lists in self.example_entity_dict.items():
            for tagged_words in value_lists:
                if type(tagged_words) is list:
                    str_tagged_words = self.tagged_words_to_str(tagged_words)
                    example_tagged_ngrams = cb.find_ngrams_by_tagged_words(self.example_ngrams, tagged_words)
                    logging.info("example_tagged_ngrams")
                    logging.info(example_tagged_ngrams)
                    example_tagged_words_ngram_vecs_dict[str_tagged_words] = \
                        [doc2vec_model.infer_vector(ngram) for ngram in example_tagged_ngrams]
        self.example_tagged_words_ngram_vecs_dict = example_tagged_words_ngram_vecs_dict

    def test_file_processing(self, test_file_path):
        super().test_file_processing(test_file_path)
        ngrams = ex_parsing.ngrams_from_file(test_file_path, self.context_size, tagged=True)
        sentences = [util.sentence_from_tagged_ngram(t) for t in ngrams]
        # logging.info(ngrams)
        logging.info("sentences: %d" % len(sentences))
        self.context_sized_test_wv_dict = self.doc_vector_to_dict_by_list(self.context_vec_model, sentences)

    def score(self, key, tagged_gram, test_file_path, wv_dict, **kwargs):
        # tagged_gram: [['esafetyworld', 'comp'], ['inc', 'end']]
        # find ngrams in test file similar to example
        example_tagged_words_ngram_vecs = \
            self.example_tagged_words_ngram_vecs_dict[self.tagged_words_to_str(tagged_gram)]
        # similar_contexts = \
        #     similar_grams_by_doc_vecs(example_tagged_words_ngram_vecs, self.context_sized_test_wv_dict)
        # logging.info('similar contexts:')
        # logging.info(similar_contexts)
        # # similar_contexts = set()
        # context_wv_dict = util.subset_dict_by_list2(wv_dict, similar_contexts)
        # logging.info('context_wv_dict:')
        # logging.info(len(context_wv_dict))
        context_wv_dict, context_similarity_dict = make_context_dict(example_tagged_words_ngram_vecs,
                                                                     self.context_sized_test_wv_dict,
                                                                     wv_dict,
                                                                     self.topn*2,
                                                                     self.context_threshold)

        gram = util.sentence_from_tagged_ngram(tagged_gram)
        return super().score(key, gram, test_file_path, context_wv_dict, context_sim_dict=context_similarity_dict)


# this model uses different doc vectors between context and words, which are trained both in ngrams and context_size
# [ 6.24901561  2.02633446] for context size 20
# total score (5.723827503944152, 1.555457416626519) of 531 in 100 files c = 10
class OneShotTestContext2(OneShotTestContext1):
    def doc_vectors_training(self):
        return OneShotTestDoc2Vec.doc_vectors_training(self)

    def context_doc_training(self):
        return cb.make_doc2vec_model_from_aaer()

    def post_training(self):
        self.context_vec_model = self.context_doc_training()
        self.make_example_tagged_words_ngram_vecs_dict(self.context_vec_model)


class OneShotTestD2vW2v(OneShotTestContext2):
    def doc_vectors_training(self):
        return cb.DocVecByWESum()


class OneShotTestWVSumWVPhrase(OneShotTestContext2):
    def context_doc_training(self):
        return cb.DocVecByWESum()

    def doc_vectors_training(self):
        return cb.PhraseVecTrigrams()


class OneShotTestWVSumWVPhraseBi(OneShotTestWVSumWVPhrase):
    def context_doc_training(self):
        return cb.DocVecByWESum()

    def doc_vectors_training(self):
        return cb.PhraseVecBigrams()


# this model uses word embeddings, then calculates the ngram similarity, instead of using doc2vec
# [ 85.34945763   7.69444444]
# [ 78.1666238    9.14444444] wv_size = 300
# [ 23.72189289   1.46636364]
class OneShotTestWVMean(OneShotTestDoc2Vec):
    def train(self):
        assert self.example_entity_dict
        self.doc_vec_model = self.doc_vectors_training()

    def doc_vectors_training(self):
        # train word vectors first
        return cb.DocVecByWEMean()


# [ 82.18312396   7.22619048]
class OneShotTestWVSum(OneShotTestWVMean):
    def doc_vectors_training(self):
        return cb.DocVecByWESum()


#
# total score [ 74.95494505   8.06666667] of 531 in 100 files
# total score (68.86923076923074, 9.4) of 531 in 100 files wv_size = 300
class OneShotTestWVWMD(OneShotTestWVMean):
    @staticmethod
    def doc_vector_to_dict_by_list(dv_model, grams):
        return {tuple(k): None for k in grams}

    def similar_grams_by_gram(self, gram, wv_dict, topn=None):
        # topn = topn if topn else self.topn
        logging.info('similar grams by gram:')
        logging.info(wv_dict.keys())
        wmd_dict = {g: self.doc_vec_model.wv_model.wmdistance(gram, g) for g in wv_dict}

        sorted_grams = util.sorted_tuples_from_dict(wmd_dict)
        # print(sorted_grams)
        # util.get_top_group(sorted_grams)
        return util.get_top_group(sorted_grams)


# 112.0228742    10.85714286:n_gram=5, context_size=10
# [ 105.63122521   10.66666667] wv_size=300
# [ 105.37120649   11.0047619 ] n, c = 5
# [ 108.78976112   11.2031746 ] c = 20
# [ 103.7195355    10.52857143] c = 100
# [ 183.28773449  127.57142857] c=10, add threshold for similarity>0.7
# [ 168.73066378  104.45396825] context similarity>0.7, word>0.4
# [ 170.15916306  139.9047619 ] context>0.8, word>0.4
# [ 177.92510823  124.68253968] context>0.75
# [ 183.53621934  132.68253968] word>0.5
# [ 35.65761836   0.56261023] with new score, both with word>0.5 & 0.4
# [ 35.73164273   0.72595799] context_sim>0.7, word>0.4
# [ 10.21021416   0.72595799] rouge1 = 0 for None
# [ 36.02193454   0.67657528] rouge1=1, topn=5
# [ 33.12867582   0.17167388] context_sim>0.8
# [ 36.61191707   0.86711844] context>0.75
# [ 36.81665178   0.92095238] topn=3
# [ 33.05873257   1.06810967]context>0.6
# [ 30.17060394   0.91711844] context>0.75, with weighted sim
# [ 36.52729225   1.16857143] size=512
# [ 35.63946563   0.71333333] size=512, trained on aaer_ex corpus
# # [ 30.08898944   0.67333333] with new similarity func
class OneShotTestContextWVMean(OneShotTestWVMean):
    def __init__(self, example_path, test_file_path_list, enable_saving=False, n_gram=5, **kwargs):
        super().__init__(example_path, test_file_path_list, enable_saving, n_gram, **kwargs)
        self.context_vec_model = None
        self.tagged_tokens = ex_parsing.tagged_tokens_from_file(self.example_path)
        self.example_entity_dict = \
            ex_parsing.entity_tagged_words_dict_from_tagged_tokens(self.tagged_tokens)
        self.example_ngrams = ex_parsing.ngrams_from_file(self.example_path, self.context_size, tagged=True)
        self.example_tagged_words_ngram_vecs_dict = {}
        self.context_sized_test_wv_dict = None

    def train(self):
        super().train()
        self.post_training()

    def context_doc_training(self):
        return self.doc_vec_model

    def post_training(self):
        self.context_vec_model = self.context_doc_training()
        self.make_example_tagged_words_ngram_vecs_dict(self.doc_vec_model)

    @staticmethod
    def tagged_words_to_str(tagged_words):
        return const.UNIQUE_DELIMITER.join(util.flatten_list(tagged_words))

    # @staticmethod
    # # returns a list of  ngrams in dv dict similar to doc vecs
    # def similar_grams_by_doc_vecs(doc_vecs, dv_dict, topn=TOP_N):
    #     # logging.info('similiar_words_by_word: word: ' + word + 'text_wv_dict: ' + str(test_wv_dict))
    #     grams_found = set()
    #     for doc_vec in doc_vecs:
    #         grams = similar_grams_by_vec(doc_vec, dv_dict, topn=topn)
    #         grams_found.update(set(tuple(gram) for gram in grams))
    #     return grams_found

    # make a dict which is convenient to look up a list of ngram vectors by tagged_words
    def make_example_tagged_words_ngram_vecs_dict(self, doc2vec_model):
        assert self.example_entity_dict
        example_tagged_words_ngram_vecs_dict = {}
        for entity, value_lists in self.example_entity_dict.items():
            for tagged_words in value_lists:
                if type(tagged_words) is list:
                    str_tagged_words = self.tagged_words_to_str(tagged_words)
                    example_tagged_ngrams = cb.find_ngrams_by_tagged_words(self.example_ngrams, tagged_words)
                    logging.info("example_tagged_ngrams")
                    logging.info(example_tagged_ngrams)
                    example_tagged_words_ngram_vecs_dict[str_tagged_words] = \
                        [doc2vec_model.infer_vector(ngram) for ngram in example_tagged_ngrams]
        self.example_tagged_words_ngram_vecs_dict = example_tagged_words_ngram_vecs_dict

    def test_file_processing(self, test_file_path):
        super().test_file_processing(test_file_path)
        ngrams = ex_parsing.ngrams_from_file(test_file_path, self.context_size, tagged=True)
        sentences = [util.sentence_from_tagged_ngram(t) for t in ngrams]
        # logging.info(ngrams)
        # logging.info(sentences)
        self.context_sized_test_wv_dict = self.doc_vector_to_dict_by_list(self.context_vec_model, sentences)

    def score(self, key, tagged_gram, test_file_path, wv_dict, **kwargs):
        # tagged_gram: [['esafetyworld', 'comp'], ['inc', 'end']]
        # find ngrams in test file similar to example
        example_tagged_words_ngram_vecs = \
            self.example_tagged_words_ngram_vecs_dict[self.tagged_words_to_str(tagged_gram)]
        context_wv_dict, context_similarity_dict = make_context_dict(example_tagged_words_ngram_vecs,
                                                                     self.context_sized_test_wv_dict,
                                                                     wv_dict,
                                                                     self.topn*2,
                                                                     self.context_threshold)

        gram = util.sentence_from_tagged_ngram(tagged_gram)
        return super().score(key, gram, test_file_path, context_wv_dict, context_sim_dict=context_similarity_dict)


# [ 35.63946563   0.71333333]
# [ 36.87764634   1.69333333] with no weighted sim func
# [ 36.36183659   1.24      ] with topn=5
# [ 37.62390453   2.08222222] with context_size = 20
class OneShotTestContextWVSum(OneShotTestContextWVMean):
    def doc_vectors_training(self):
        return cb.DocVecByWESum()


# total score (65.30256410256409, 3.733333333333333) of 531 in 100 files
# [ 52.06866174   3.28738739] for context_size = 20
# [ 62.4800781    2.73333333] c = 40
# [ 84.52690349   5.73333333] c = 100
# [ 97.0596678   15.83333333] c = 200
#  91.70728684  16.16666667] c = 200, wv_size = 300
class OneShotTestContext4(OneShotTestContextWVMean, OneShotTestWVWMD):
    def score(self, key, tagged_gram, test_file_path, wv_dict, **kwargs):
        # tagged_gram: [['esafetyworld', 'comp'], ['inc', 'end']]
        # find ngrams in test file similar to example
        similar_contexts = \
            self.similar_grams_by_gram([g[0] for g in tagged_gram],
                                       self.context_sized_test_wv_dict,
                                       topn=self.topn)
        logging.info('similar contexts:')
        print(similar_contexts)
        # similar_contexts = set()
        context_wv_dict = util.subset_dict_by_list2(wv_dict, similar_contexts)
        logging.info('context_wv_dict:')
        logging.info(len(context_wv_dict))

        gram = util.sentence_from_tagged_ngram(tagged_gram)
        return OneShotTestDoc2Vec.score(self, key, gram, test_file_path, context_wv_dict)


class OneShotTestContext5(OneShotTestContext4):
    @staticmethod
    def doc_vector_to_dict_by_list(dv_model, grams):
        OneShotTestWVMean.doc_vector_to_dict_by_list(dv_model, grams)

    def score(self, key, tagged_gram, test_file_path, wv_dict, **kwargs):
        # tagged_gram: [['esafetyworld', 'comp'], ['inc', 'end']]
        # find ngrams in test file similar to example
        example_tagged_words_ngram_vecs = \
            self.example_tagged_words_ngram_vecs_dict[self.tagged_words_to_str(tagged_gram)]
        # print(self.context_sized_test_wv_dict)
        # similar_contexts = \
        #     similar_grams_by_doc_vecs(example_tagged_words_ngram_vecs, self.context_sized_test_wv_dict)
        # logging.info('similar contexts:')
        # print(similar_contexts)
        # # similar_contexts = set()
        # context_wv_dict = util.subset_dict_by_list2(wv_dict, similar_contexts)
        # logging.info('context_wv_dict:')
        # logging.info(len(context_wv_dict))
        context_wv_dict, context_similarity_dict = make_context_dict(example_tagged_words_ngram_vecs,
                                                                     self.context_sized_test_wv_dict,
                                                                     wv_dict,
                                                                     self.topn*2,
                                                                     self.context_threshold)

        gram = util.sentence_from_tagged_ngram(tagged_gram)
        return OneShotTestDoc2Vec.score(self, key, gram, test_file_path, context_wv_dict)


# [ 20.60965956   6.07702742]
# [ 20.65537384   6.13036075]
# [ 20.65537384   6.13036075]
class OneShotTestT2TNoContext(OneShotTestWVMean):
    def doc_vectors_training(self):
        return dl_context.T2TContextModel(load_aaer_test_data=True, doc_length=self.n, one_to_n=True)

    @staticmethod
    def doc_vector_to_dict_by_list(dv_model, grams):
        assert isinstance(dv_model, dl_context.T2TContextModel)
        return dv_model.infer_vectors_dict(grams)


# 59.52085954  15.43333333 for c=10
# [ 59.20181192  16.6 ] for c=40 (c=100 shows little difference)
# trained with ngram = 10 aaer corpus TARGET_SIZE = 5 WINDOW_SIZE = 2
# no good from target_size=10, ngram=20
# trained on ngram=5
# [ 77.51556777  24.2       ] for c=10
# [ 63.7470764    2.57505241] ... for c=10, after more training
# [ 64.94576271   2.17505241] for c=5
# [ 56.76652084   2.17505241][ 56.86652084   2.17505241] c=20
# above training based on predict training:target = n_grams[i + window_size][-target_size:]
# below will be based on generating training: target = window + source + window
#  93.69231741  17.77777778 c=20, window=3, target ngram = 10
#  (97.30374257080713, 22.019047619047623) c=10, topn=1
# (171.97229437229439, 121.36666666666669) c=10 with overall similarity threshold=0.7
# [ 169.02065344  102.66666667] context similarity>0.75, words>0.4
# [ 34.24517008   1.21718484] new score, context>0.8
# [ 25.77144166   3.63803705] context>0.6
# [ 29.11031324   3.16230046] context>0.7
# [ 31.65965501   3.4780464 ] context>0.75
# [ 34.43088436   1.05051817] context>0.8
# [ 32.39227328   3.30161616] context>0.75, topn=3
# above trained on: N_GRAMS = 10 WINDOW_SIZE = 3
# ------------------
# below trained on: window_size=2
# [ 29.61410367   2.81287157]
# below trained on ngrams=20, window size=3
# (17.514451065730135, 1.0966666666666667)
# ngrams=14, window size=2:
# [ 33.99083341   1.11666667]
# [ 32.7550938    0.17333333]
# ngrams=14, window size=3, trained on extra corpus:
# [ 22.84718314   3.61809524]
# window size=2:
# [ 25.10924964   3.31380952]
# [ 33.27467199   3.85131313] with extra corpus & revised similarity weighting func
# [ 0.24482222  0.02278912] with context_size 20, new trained
class OneShotTestT2TModel(OneShotTestContext1):
    @staticmethod
    def doc_vector_to_dict_by_list(dv_model, grams):
        assert isinstance(dv_model, dl_context.T2TContextModel)
        return dv_model.infer_vectors_dict(grams)

    def doc_vectors_training(self):
        return dl_context.T2TContextModel(load_aaer_test_data=True, doc_length=self.context_size)

    # make a dict which is convenient to look up a list of ngram vectors by tagged_words
    def make_example_tagged_words_ngram_vecs_dict(self, doc2vec_model):
        assert self.example_entity_dict
        temp_dict = {}
        all_example_tagged_ngrams = []
        for entity, value_lists in self.example_entity_dict.items():
            for tagged_words in value_lists:
                if type(tagged_words) is list:
                    str_tagged_words = self.tagged_words_to_str(tagged_words)
                    example_tagged_ngrams = [' '.join(ngram) for ngram in
                                             cb.find_ngrams_by_tagged_words(self.example_ngrams, tagged_words)]
                    temp_dict[str_tagged_words] = example_tagged_ngrams
                    all_example_tagged_ngrams += example_tagged_ngrams
                    #     [doc2vec_model.infer_vector(ngram) for ngram in all_example_tagged_ngrams]
        # to infer all ngrams at one time, for the sake of seed
        doc2vec_model.update(all_example_tagged_ngrams)

        for word in temp_dict.keys():
            self.example_tagged_words_ngram_vecs_dict[word] = \
                [doc2vec_model.infer_vector(ngram) for ngram in temp_dict[word]]


# this class use t2t model to infer context, while wv mean to infer grams(entities)
# [ 12.31577938   0.72539683] rouge1=0 for None similarity>0.7, words>0.4
# [ 25.96934622   1.32142857] context>0.75
# [ 33.1232684    0.67047619] trained on ngrams=20, window size=3
# [ 35.74163226   0.51333333] trained on ngrams=14, window size=2
# [ 24.19922422   1.71333333] ngrams=14, window size=3, trained on extra corpus:
class OneShotTestT2TWVMean(OneShotTestContextWVMean):
    @staticmethod
    def context_vector_to_dict_by_list(dv_model, grams):
        assert isinstance(dv_model, dl_context.T2TContextModel)
        return dv_model.infer_vectors_dict(grams)

    def context_doc_training(self):
        return dl_context.T2TContextModel(load_aaer_test_data=True, doc_length=self.context_size)

    def test_file_processing(self, test_file_path):
        OneShotTestDoc2Vec.test_file_processing(self, test_file_path)
        ngrams = ex_parsing.ngrams_from_file(test_file_path, self.context_size, tagged=True)
        sentences = [util.sentence_from_tagged_ngram(t) for t in ngrams]
        # logging.info(ngrams)
        # logging.info(sentences)
        self.context_sized_test_wv_dict = self.context_vector_to_dict_by_list(self.context_vec_model, sentences)

    # make a dict which is convenient to look up a list of ngram vectors by tagged_words
    def make_example_tagged_words_ngram_vecs_dict(self, doc2vec_model):
        assert self.example_entity_dict
        temp_dict = {}
        all_example_tagged_ngrams = []
        for entity, value_lists in self.example_entity_dict.items():
            for tagged_words in value_lists:
                if type(tagged_words) is list:
                    str_tagged_words = self.tagged_words_to_str(tagged_words)
                    example_tagged_ngrams = [' '.join(ngram) for ngram in
                                             cb.find_ngrams_by_tagged_words(self.example_ngrams, tagged_words)]
                    temp_dict[str_tagged_words] = example_tagged_ngrams
                    all_example_tagged_ngrams += example_tagged_ngrams
                    #     [doc2vec_model.infer_vector(ngram) for ngram in all_example_tagged_ngrams]
        # to infer all ngrams at one time, for the sake of seed
        doc2vec_model.infer_vectors_dict(all_example_tagged_ngrams)

        for word in temp_dict.keys():
            self.example_tagged_words_ngram_vecs_dict[word] = \
                [doc2vec_model.infer_vector(ngram) for ngram in temp_dict[word]]

    def post_training(self):
        self.context_vec_model = self.context_doc_training()
        self.make_example_tagged_words_ngram_vecs_dict(self.context_vec_model)


class OneShotTestT2TWVSum(OneShotTestT2TWVMean):
    def doc_vectors_training(self):
        return cb.DocVecByWESum()


class OneShotTestT2TWVPhrase(OneShotTestT2TWVMean):
    def doc_vectors_training(self):
        return cb.PhraseVecTrigrams()


# this class only scores context similarity
class ContextTest(OneShotTestContext1):
    def doc_vectors_training(self):
        return None

    def make_test_wv_dict(self, test_grams):
        self.test_wv_dict = None

    def context_doc_training(self):
        raise NotImplementedError

    @staticmethod
    def doc_vector_to_dict_by_list(dv_model, grams):
        raise NotImplementedError

    def post_training(self):
        self.context_vec_model = self.context_doc_training()
        self.make_example_tagged_words_ngram_vecs_dict(self.context_vec_model)

    def score(self, key, tagged_gram, test_file_path, wv_dict, **kwargs):
        example_tagged_words_ngram_vecs = \
            self.example_tagged_words_ngram_vecs_dict[self.tagged_words_to_str(tagged_gram)]
        context_sim_dict = make_context_sim_dict(example_tagged_words_ngram_vecs,
                                                 self.context_sized_test_wv_dict,
                                                 self.topn*2,
                                                 self.context_threshold)

        gram = util.sentence_from_tagged_ngram(tagged_gram)

        # print(wv_dict.keys())

        contexts_found = list(context_sim_dict.keys())
        print('Contexts found:')
        print(contexts_found)
        print('correct:')

        # print(gram)
        example_tagged_ngrams = cb.find_ngrams_by_tagged_words(self.example_ngrams, tagged_gram)
        print(example_tagged_ngrams)
        # if len(contexts_found) == 0:
        #     contexts_found = []
        # else:
        #     contexts_found = [contexts_found[0]]
        # scores, counts = score_by_rouge(contexts_found, self.test_entity_dict, key)
        score = (0, 0)
        count = 0
        if tagged_gram:
            for context in contexts_found:
                # print(words_found)
                score = util.tuple_add(score, (rouge.rouge_1(context, example_tagged_ngrams, alpha=0.5),
                                               rouge.rouge_2(context, example_tagged_ngrams, alpha=0.5)))
                # todo: do we need to +=1? it seems not accurate
                count += 1
                # print(score)
        elif not contexts_found:  # both do not have similar words compared to example
            score = util.tuple_add(score, (1, 0))  # set rouge2 as 0 because for single word rouge2 returns 0
            count = 1
        print("rouge:", score)
        self.score_dict[test_file_path] = util.tuple_add(self.score_dict[test_file_path], score)
        return count


# [ 18.39388458   4.56925512]
class ContextTestWVMean(ContextTest):
    @staticmethod
    def doc_vector_to_dict_by_list(dv_model, grams):
        return OneShotTestWVMean.doc_vector_to_dict_by_list(dv_model, grams)

    def context_doc_training(self):
        return cb.DocVecByWEMean()


# [ 13.34222989   2.66634493]
# [ 0.13516291  0.01667692]
class ContextTestT2T(ContextTest):
    def context_doc_training(self):
        return dl_context.T2TContextModel(load_aaer_test_data=True, doc_length=self.context_size)

    @staticmethod
    def doc_vector_to_dict_by_list(dv_model, grams):
        return OneShotTestT2TModel.doc_vector_to_dict_by_list(dv_model, grams)


# [ 19.07455124   5.20555142]
class ContextTestWVSum(ContextTestWVMean):
    def context_doc_training(self):
        return cb.DocVecByWESum()


class ContextTestDoc2vec(ContextTest):
    @staticmethod
    def doc_vector_to_dict_by_list(dv_model, grams):
        return OneShotTestDoc2Vec.doc_vector_to_dict_by_list(dv_model, grams)

    def context_doc_training(self):
        return cb.make_doc2vec_model_from_aaer()
