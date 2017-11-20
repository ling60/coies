import model_testing.oneshot_test as oneshot
import model_testing.dl_context_models as dl_models
import t2t_make_data_files
import common.file_tools as ft
import common.constants as const
import os
import csv
import numpy as np
import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)


def run_for_epochs(example_path, files, model_class, config_dict, enable_saving=False, epochs=1):
    score_arr = np.array([0.0, 0.0])
    one_shot_test = None
    if not os.path.exists(const.RESULTS_DIR):
        os.makedirs(const.RESULTS_DIR)
    file_path = os.path.join(const.RESULTS_DIR, model_class.__name__)
    for epoch in range(0, epochs):
        # one_shot_test = model_class(example_path, files, enable_saving=enable_saving, context_size=10)
        one_shot_test = model_class(example_path, files, enable_saving=enable_saving, conf_dict=config_dict)
        one_shot_test.train()
        score, _ = one_shot_test.test()
        score_arr = np.add(score_arr, score)
    avg_score = np.divide(score_arr, epochs * len(files))
    print(avg_score)

    with open(file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if one_shot_test:
            try:
                if type(one_shot_test.context_vec_model) is dl_models.T2TContextModel \
                        or type(one_shot_test.doc_vec_model) is dl_models.T2TContextModel:
                    conf_dict = t2t_make_data_files.load_configs()
                    csv_writer.writerow([config_dict, avg_score, conf_dict])
                else:
                    csv_writer.writerow([config_dict, avg_score])
            except AttributeError:
                csv_writer.writerow([config_dict, avg_score])
    return avg_score


# todo: set validation size: 49:50?
def test(example_path, files, model_class, enable_saving=False, epochs=1):
    config_dict = oneshot.base_conf_dict
    run_for_epochs(example_path, files, model_class, config_dict=config_dict, enable_saving=enable_saving, epochs=epochs)


def grid_conf_dict_generator():
    config_dict = oneshot.base_conf_dict
    assert config_dict['topn']
    for topn in range(3, 7):
        config_dict['topn'] = topn
        for i in range(60, 100, 5):
            assert config_dict['context_threshold']
            config_dict['context_threshold'] = i / 100
            for n in range(4, 9):
                assert config_dict['word_threshold']
                config_dict['word_threshold'] = n / 10
                for context_size in [100]:
                    assert config_dict['context_size']
                    config_dict['context_size'] = context_size
                    yield config_dict


def grid_search(example_path, model_class, enable_saving=True, epochs=1):
    files = ft.list_file_paths_under_dir(const.VALIDATION_DIR, ['txt'])

    for conf in grid_conf_dict_generator():
        run_for_epochs(example_path, files, model_class, config_dict=conf,
                       enable_saving=enable_saving, epochs=epochs)


file_list = ft.list_file_paths_under_dir(const.TEST_DIR, ['txt'])
# file_list = [os.path.join(const.TEST_DIR, '34-71576.txt')]
# conf_dict = oneshot.base_conf_dict
# conf_dict['context_size'] = 100
# conf_dict['context_threshold'] = 0.9
# conf_dict['word_threshold'] = 0.6
# conf_dict['topn'] = 5
# run_for_epochs(const.EXAMPLE_FILE, file_list, oneshot.OneShotTestWVSumWVPhraseBi, config_dict=conf_dict, epochs=1)

grid_search(const.EXAMPLE_FILE, oneshot.OneShotTestWVSumWVPhrase)

