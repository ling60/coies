# This lib contains common tools for file/path access
import os
import re
import string
import logging
import common.constants as const


def check_dir_ending(str_dir):
    str_dir = str_dir.strip()
    return str_dir if str_dir[-1] == '/' else str_dir + '/'


def make_regex_by_file_extensions(file_extensions):
    assert type(file_extensions) is list
    return "^(.*\.(" + "|".join(file_extensions) + ")$)?$"


# return list of filenames given dir and a list of extensions
def list_files_under_dir(dir_path, file_extensions):
    dir_path = check_dir_ending(dir_path)
    regext = make_regex_by_file_extensions(file_extensions)
    file_list = [f for f in os.listdir(dir_path) if re.match(regext, f)]
    return file_list


# return list of absolute paths given dir and extensions
def list_file_paths_under_dir(dir_path, file_extensions):
    dir_path = check_dir_ending(dir_path)
    regext = make_regex_by_file_extensions(file_extensions)
    file_list = [os.path.join(dir_path + f) for f in os.listdir(dir_path) if re.match(regext, f)]
    return file_list


# get file name without extension from file path
def file_name_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


# convert text in bytes to readable string
def messy_codec_handling(b_text):
    assert type(b_text) is bytes
    text = b_text.decode('ascii', errors='ignore')
    # print(text)
    return text


def messy_codec_file_to_text(file_path):
    with open(file_path, 'rb') as f:
        text = f.read()
    return messy_codec_handling(text)


def remove_punctuation_from_tokens(tokens):
    assert type(tokens) is list
    cleaned_tokens = []
    translate_table = dict((ord(char), None) for char in string.punctuation)
    for token in tokens:
        token = token.translate(translate_table)
        if token:
            cleaned_tokens.append(token)
    return cleaned_tokens


# our own tokenizer, removing spaces and punctuations, then return the lower case of tokens
def text_tokenizer(text):
    delimiters = [' ', '\n', '\r', '\t', '\v', '\f', '\0'] + list(string.punctuation)
    regex = '|'.join(map(re.escape, delimiters))
    tokens = re.split(regex, text)
    tokens[:] = [token.strip() for token in tokens if token.strip() != '']
    return tokens


def text_to_sentences(text):
    delimiters = ['.', '\n', '\r']
    regex = '|'.join(map(re.escape, delimiters))
    sentences = re.split(regex, text)
    sentences[:] = [token.strip() for token in sentences if token.strip() != '']
    return sentences


# filter word2vec/fasttext .vec file given a vocab set
def filter_vec_file_by_set(vec_file_path, vocab_set, output_file_path):
    output_list = [" "]  # create the output list with blank header
    with open(vec_file_path, 'r', encoding='utf-8') as vec_f:
        header = next(vec_f)   # skipping the header of vec file
        output_list[0] += header.split(' ')[1]  # copy the number of dimensions to the output header
        for line in vec_f:
            word_vec = line.split(' ')
            word = word_vec[0]
            if word in vocab_set:
                output_list.append(line)

    words_count = len(output_list) - 1
    output_list[0] = str(words_count) + output_list[0]

    with open(output_file_path, 'w') as output_f:
        output_f.writelines(output_list)
    logging.info(output_file_path + ' created')


# find source file with the same name of given file in example dir
def get_source_file_by_example_file(example_path):
    file_name = os.path.basename(example_path)
    return os.path.join(const.DATA_PATH, const.AAER_PATH, file_name)
