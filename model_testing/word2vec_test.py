import gensim
import text_cleaning.example_parsing_no_position as ex_parsing
import common.constants as const
import common.file_tools as ft
import common.utilities as util
import os
import logging


def word_vectors_from_file(file_name):
    tokens, entity_dict = ex_parsing.parse_file(const.DATA_PATH + file_name)
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = gensim.models.Word2Vec(tokens, min_count=1)
    return model, entity_dict


def word_vectors_from_tokens(tokens):
    return gensim.models.Word2Vec(tokens, min_count=1)


def remove_punctuations_from_entity_dict(entity_dict):
    for k, value_list in entity_dict.items():
        value_list[:] = [ft.text_tokenizer(value) for value in value_list]
        entity_dict[k] = value_list
    return entity_dict


def score(words_found, test_entity_dict, entity_key):
    hits = 0
    if entity_key in test_entity_dict:
        for word in words_found:
            if word in sum(test_entity_dict[entity_key], []):
                hits += 1
    return hits


def one_shot_test(example_path, test_file_path_list):
    score_dict = {}

    example_tokens, example_entity_dict = ex_parsing.parse_file(example_path)
    if example_entity_dict:
        # do the test
        example_model = word_vectors_from_tokens(example_tokens)
        example_entity_dict = remove_punctuations_from_entity_dict(example_entity_dict)
        print(example_entity_dict)

        for test_file_path in test_file_path_list:
            score_dict[test_file_path] = 0
            test_tokens, test_entity_dict = ex_parsing.parse_file(test_file_path)
            test_entity_dict = remove_punctuations_from_entity_dict(test_entity_dict)
            test_model = word_vectors_from_tokens(test_tokens)

            for k, value_list in example_entity_dict.items():
                for value in value_list:
                    if type(value) is list:
                        for v in value:
                            # find most similar words in test file
                            print('similar to:' + v)
                            words_found = [word[0] for word in
                                           test_model.similar_by_vector(example_model.wv[v], topn=1)]
                            hits = score(words_found, test_entity_dict, k)
                            print("hits:", hits)
                            score_dict[test_file_path] += hits

                    else:
                        raise Exception('the value of entity dict should be list')
        print(score_dict)
    print('total score', sum(score_dict.values()), 'of', len(score_dict.keys()), 'files')


def word_vec_from_file_list(file_path_list):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    total_tokens = []
    for file in file_path_list:
        tokens, _ = ex_parsing.parse_file(file)
        # print(tokens)
        total_tokens.extend(tokens)
    model = gensim.models.Word2Vec(total_tokens, min_count=1)
    return model


def one_shot_test1(example_path, test_file_path_list):
    score_dict = {}

    example_tokens, example_entity_dict = ex_parsing.parse_file(example_path)
    if example_entity_dict:
        # train the base model from all files
        model = word_vec_from_file_list(test_file_path_list)
        # model.train(example_tokens, total_examples=model.corpus_count)
        # do the test
        example_entity_dict = remove_punctuations_from_entity_dict(example_entity_dict)
        print(example_entity_dict)

        for test_file_path in test_file_path_list:
            score_dict[test_file_path] = 0
            test_tokens, test_entity_dict = ex_parsing.parse_file(test_file_path)
            test_entity_dict = remove_punctuations_from_entity_dict(test_entity_dict)
            # model.train(test_tokens, total_examples=model.corpus_count)
            sub_word_vector_dict = util.word_vector_to_dict_by_list(model.wv, util.flatten_list(test_tokens))
            for k, value_list in example_entity_dict.items():
                for value in value_list:
                    if type(value) is list:
                        for v in value:
                            # find most similar words in test file
                            print('similar to:' + v)
                            words_found = [word[0] for word in
                                           util.similar_by_vector(model.wv[v],
                                                                  sub_word_vector_dict, topn=1)]
                            hits = score(words_found, test_entity_dict, k)
                            print("hits:", hits)
                            score_dict[test_file_path] += hits

                    else:
                        raise Exception('the value of entity dict should be list')
        print(score_dict)
    print('total score', sum(score_dict.values()), 'of', len(score_dict.keys()), 'files')


def one_shot_test2(example_path, test_file_path_list):
    score_dict = {}

    example_tokens, example_entity_dict = ex_parsing.parse_file(example_path)
    if example_entity_dict:
        # train the base model from all files
        model = word_vec_from_file_list(ft.list_file_paths_under_dir(const.DATA_PATH, ['txt']))
        # model.train(example_tokens, total_examples=model.corpus_count)
        # do the test
        example_entity_dict = remove_punctuations_from_entity_dict(example_entity_dict)
        print(example_entity_dict)

        for test_file_path in test_file_path_list:
            score_dict[test_file_path] = 0
            test_tokens, test_entity_dict = ex_parsing.parse_file(test_file_path)
            test_entity_dict = remove_punctuations_from_entity_dict(test_entity_dict)
            # model.train(test_tokens, total_examples=model.corpus_count)
            sub_word_vector_dict = util.word_vector_to_dict_by_list(model.wv, util.flatten_list(test_tokens))
            for k, value_list in example_entity_dict.items():
                for value in value_list:
                    if type(value) is list:
                        for v in value:
                            # find most similar words in test file
                            # print('similar to:' + v)
                            words_found = [word[0] for word in
                                           util.similar_by_vector(model.wv[v],
                                                                  sub_word_vector_dict, topn=1)]
                            hits = score(words_found, test_entity_dict, k)
                            # print("hits:", hits)
                            score_dict[test_file_path] += hits

                    else:
                        raise Exception('the value of entity dict should be list')
        print(score_dict)
    print('total score', sum(score_dict.values()), 'of', len(score_dict.keys()), 'files')

# one_shot_test("examples/"+"34-53330.txt", ["examples/"+"34-53391.txt"])
# file_list = ft.list_files_under_dir(const.DATA_PATH + 'examples', ['txt'])
sub_dir = 'examples/'
file_list = ft.list_file_paths_under_dir(os.path.join(const.DATA_PATH, sub_dir), ['txt'])
one_shot_test2(os.path.join(const.TEST_DIR, '34-53330.txt'),
              [file_path for file_path in file_list])
