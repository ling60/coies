import model_testing.oneshot_test as oneshot
import common.file_tools as ft
import common.constants as const
import text_cleaning.example_parsing as ex_parsing
import numpy as np
import logging
import tensorflow as tf


logging.basicConfig(level=logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)


def test(example_path, files, model_class, enable_saving=False, epochs=1):
    score_arr = np.array([0.0, 0.0])
    for epoch in range(0, epochs):
        one_shot_test = model_class(example_path, files, enable_saving=enable_saving, context_size=20)
        one_shot_test.train()
        score, _ = one_shot_test.test()
        score_arr = np.add(score_arr, score)
    print(np.divide(score_arr, epochs))


file_list = ft.list_file_paths_under_dir(const.TEST_DIR, ['txt'])

test(const.EXAMPLE_FILE, file_list, oneshot.OneShotTestT2TModel, epochs=1)
# one_shot_test = oneshot.OneShotTestRandom(const.DATA_PATH + 'examples/34-53330.txt', file_list, enable_saving=True)

# print(ex_parsing.tokens_from_file(example_file_path))

# for path in ft.list_file_paths_under_dir(const.HUMAN_DIR, ['txt']):
#     print(path)
#     tagged_tokens = ex_parsing.tagged_tokens_from_file(path)
#     entities = ex_parsing.entity_dict_from_tagged_tokens(tagged_tokens)
#     print(entities)
