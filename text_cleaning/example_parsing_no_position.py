# This file uses standard procedure to parse entities marked by example files
# results would be {comp:["DELL","GE"...],
#                   date:["2008",..], item:...}
# position information thus will be lost!
# and nltk tokenized sentences:[["sent","1"], ["sent", "2"]..]

import nltk
from html.parser import HTMLParser
import common.constants as const
import common.file_tools as ft
import common.utilities as util
import logging


ending_mark_length = len('<' + const.TAG_ENDING + '>')


class MyExampleParser(HTMLParser):
    def error(self, message):
        pass

    def __init__(self):
        super().__init__()
        self.__found_tag = None
        self.entity_dict = {}

    def handle_starttag(self, tag, attrs):
        if tag[-1] == const.TAG_POSTFIX:
            self.__found_tag = tag[:-1]

    def handle_endtag(self, tag):
        self.__found_tag = None

    def handle_data(self, data):
        if self.__found_tag is not None:
            if self.__found_tag in self.entity_dict:
                self.entity_dict[self.__found_tag].append(data)
            else:
                self.entity_dict[self.__found_tag] = [data]
        self.__found_tag = None


def text_tokenizer(text):
    sentences = nltk.sent_tokenize(text)
    tokens = [ft.remove_punctuation_from_tokens(nltk.word_tokenize(sentence)) for sentence in sentences]
    return tokens


def parse_file(file_path):
    with open(file_path, 'rb') as f:
        text = f.read()
        # text = str(text, 'latin-1')
    text = ft.messy_codec_handling(text).lower()
    parser = MyExampleParser()
    parser.feed(text)
    entity_dict = parser.entity_dict
    # print(entity_dict)
    my_tokens = text_tokenizer(text)
    # print(my_tokens)
    return my_tokens, entity_dict


# combine text files under a dir to one file, without punctuations
def dir_to_file_without_punctuations(dir_path, extension='txt', file_name=False):
    file_names = ft.list_file_paths_under_dir(dir_path, [extension])
    tokens = []
    for fname in file_names:
        temp_tokens, _ = parse_file(fname)
        tokens.extend(util.flatten_list(temp_tokens))

    if not file_name:
        file_name = '_'.join(dir_path.split('/')[-2:])
    with open(file_name, 'w') as f:
        logging.info('saving to:', file_name)
        f.write(' '.join(tokens))

# dir_to_file_without_punctuations(const.DATA_PATH + "sec/aaer/")
