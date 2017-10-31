import model_testing.oneshot_test as oneshot
import common.file_tools as ft
import common.constants as const
import text_cleaning.example_parsing as ex_parsing
import os
import csv
import numpy as np
import copy
import logging
import tensorflow as tf


logging.basicConfig(level=logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)


def test(example_path, files, model_class, enable_saving=False, epochs=1):
    score_arr = np.array([0.0, 0.0])
    conf_dict = oneshot.base_conf_dict
    conf_dict_list = []
    if not os.path.exists(const.RESULTS_DIR):
        os.makedirs(const.RESULTS_DIR)
    file_path = os.path.join(const.RESULTS_DIR, model_class.__name__)

    with open(file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        assert conf_dict['topn']
        for topn in range(1, 10):
            conf_dict['topn'] = topn
            for i in range(50, 100, 5):
                assert conf_dict['context_threshold']
                conf_dict['context_threshold'] = i/100
                for n in range(2, 7):
                    assert conf_dict['word_threshold']
                    conf_dict['word_threshold'] = n/10
                    for context_size in range(10, 110, 10):
                        assert conf_dict['context_size']
                        conf_dict['context_size'] = context_size
                        conf_dict_list.append(copy.deepcopy(conf_dict))
        for conf in conf_dict_list:
            for epoch in range(0, epochs):
                # one_shot_test = model_class(example_path, files, enable_saving=enable_saving, context_size=10)
                one_shot_test = model_class(example_path, files, enable_saving=enable_saving, conf_dict=conf)
                one_shot_test.train()
                score, _ = one_shot_test.test()
                score_arr = np.add(score_arr, score)
            avg_score = np.divide(score_arr, epochs)
            print(avg_score)
            csv_writer.writerow([conf_dict, avg_score])


file_list = ft.list_file_paths_under_dir(const.TEST_DIR, ['txt'])
# file_list = [os.path.join(const.TEST_DIR, '34-71576.txt')]
test(const.EXAMPLE_FILE, file_list, oneshot.OneShotTestContextWVSum, epochs=1)
# one_shot_test = oneshot.OneShotTestRandom(const.DATA_PATH + 'examples/34-53330.txt', file_list, enable_saving=True)

# print(ex_parsing.tokens_from_file(example_file_path))

# for path in ft.list_file_paths_under_dir(const.HUMAN_DIR, ['txt']):
#     print(path)
#     tagged_tokens = ex_parsing.tagged_tokens_from_file(path)
#     entities = ex_parsing.entity_dict_from_tagged_tokens(tagged_tokens)
#     print(entities)
