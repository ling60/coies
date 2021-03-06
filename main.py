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
import time


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
    str_datetime = time.strftime("%Y-%m-%d %H:%M")
    result_list = [config_dict, avg_score, str_datetime, 'aaer_ex']
    with open(file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if one_shot_test:
            try:
                if type(one_shot_test.context_vec_model) is dl_models.T2TContextModel \
                        or type(one_shot_test.doc_vec_model) is dl_models.T2TContextModel:
                    conf_dict = t2t_make_data_files.load_configs()
                    result_list.append(conf_dict)
                    csv_writer.writerow(result_list)
                else:
                    csv_writer.writerow(result_list)
            except AttributeError:
                csv_writer.writerow(result_list)
    return avg_score


# todo: set validation size: 49:50?
def test(example_path, files, model_class, enable_saving=False, epochs=1):
    config_dict = oneshot.base_conf_dict
    run_for_epochs(example_path, files, model_class, config_dict=config_dict, enable_saving=enable_saving, epochs=epochs)


def grid_conf_dict_generator():
    config_dict = oneshot.base_conf_dict
    for context_size in range(10, 200, 10):
        assert config_dict['context_size']
        config_dict['context_size'] = context_size
        yield config_dict


def grid_search(example_path, model_class, enable_saving=True, epochs=1):
    files = ft.list_file_paths_under_dir(const.VALIDATION_DIR, ['txt'])

    for conf in grid_conf_dict_generator():
        run_for_epochs(example_path, files, model_class, config_dict=conf,
                       enable_saving=enable_saving, epochs=epochs)


def validate_with_more(model):
    file_list = ft.list_file_paths_under_dir(const.TEST_DIR, ['txt'])
    # file_list = [os.path.join(const.TEST_DIR, '34-71576.txt')]
    conf_dict = oneshot.base_conf_dict
    # example_file2 = const.VALIDATION_DIR + '/34-43389.txt'

    scores = []

    for validate_file in ft.list_file_paths_under_dir(const.VALIDATION_DIR, ['txt']):
        entity_dict = oneshot.get_entity_dict_from_file(validate_file)
        if type(entity_dict) is dict:
            if len(entity_dict.keys()) > 2:
                print(validate_file)
                scores.append(run_for_epochs(validate_file, file_list, model, config_dict=conf_dict, epochs=1))
    print(scores)
    print(sum(scores)/len(scores))


# models = [oneshot.OneShotTestWVSumWVPhraseBi]
# for m in models:
#     validate_with_more(m)
file_list = ft.list_file_paths_under_dir(const.TEST_DIR, ['txt'])
models = [oneshot.OneShotTestWVSumWVPhraseBi]
for m in models:
    run_for_epochs(const.EXAMPLE_FILE, file_list, m, config_dict=oneshot.base_conf_dict, epochs=1)

for conf in grid_conf_dict_generator():
    run_for_epochs(const.EXAMPLE_FILE, file_list, oneshot.OneShotTestWVSumWVPhraseBi, config_dict=conf,
                   enable_saving=False, epochs=1)
