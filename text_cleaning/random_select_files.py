# randomly select files given dir and number, and copy to another dir
import random
import shutil
import common.file_tools as file_tools
import common.constants as const
import os


def random_select(from_dir, to_dir, sample_number, file_extensions):
    file_names = random.sample(file_tools.list_files_under_dir(from_dir, file_extensions), sample_number)
    for file_name in file_names:
        shutil.copy2(os.path.join(from_dir, file_name), to_dir)


def copy_source_of_example_files(from_dir, to_dir, example_dir):
    file_names = file_tools.list_files_under_dir(example_dir, const.TEXT_EXTENSIONS)
    to_dir = file_tools.check_dir_ending(to_dir)
    for file_name in file_names:
        shutil.copy2(os.path.join(from_dir, file_name), to_dir)


# copy_source_of_example_files(const.DATA_PATH + const.AAER_PATH,
# const.DATA_PATH + 'examples2/', const.DATA_PATH + 'examples/')
# if __name__ == "__main__":
#     if not os.path.exists(const.VALIDATION_DIR):
#         os.makedirs(const.VALIDATION_DIR)
#     random_select(const.TEST_DIR, const.VALIDATION_DIR, 49, ['txt'])
