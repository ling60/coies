# parsing entities marked by example files while keeping position info. No NLTK functions involved.
# results would be a list of tagged tokens, as: [['International', 'comp.'], ['Business', 'in'], ['Machines', 'end']]

import common.constants as const
import common.file_tools as ft
import common.utilities as utils
import logging

ending_mark_length = len('<' + const.TAG_ENDING + '>')


def replace_tag_marks_with_chars(text):
    assert type(text) is str
    text = text.replace(const.STARTING_TAGS, const.REPLACED_STARTING_TAGS)
    text = text.replace(const.ENDING_TAGS, const.REPLACED_ENDING_TAGS)
    return text


# deprecated, use tokens_to_tagged_tokens2
def tokens_to_tagged_tokens1(tokens):
    logging.warning("tokens_to_tagged_tokens1 is deprecated")
    # create a list of lists as [[token, tag], ...]
    tagged_tokens = []
    current_tag = const.NONE_TAG
    beginning_tag_pos = -1
    for pos, token in enumerate(tokens):
        # token = ord(token)
        tagged_tokens.append([token, current_tag])
        # check if any tag_marks (<>) exits
        if token == '>' and tokens[pos - 2] == '<':
            for i in range(pos - 2, pos + 1):
                # mark the tag mark as ['<', 'tag']...
                tagged_tokens[i][1] = const.MARK_TAG
            pre_token = tokens[pos - 1]
            if pre_token != const.TAG_ENDING:  # where tag begins: <comp.>
                current_tag = pre_token
                beginning_tag_pos = pos + 1
            else:  # this is where the previous tag ends: </>
                current_tag = const.NONE_TAG
                # if it is the ending tag, then change tags of tokens in the middle to 'in',
                # in the end to 'end' For example:
                # [['International', 'comp.'], ['Business', 'in'], ['Machines', 'end']]

                if beginning_tag_pos != -1 and pos > (beginning_tag_pos + ending_mark_length):
                    for m in range(beginning_tag_pos + 1, pos - ending_mark_length):
                        tagged_tokens[m][1] = const.IN_TAG
                    tagged_tokens[pos - ending_mark_length][1] = const.END_TAG
                beginning_tag_pos = -1

    # filter out tag marks like </>
    tagged_tokens[:] = [tagged_token for tagged_token in tagged_tokens if tagged_token[1] != const.MARK_TAG]
    return tagged_tokens


# parsing functions after replacing html like tags
def tokens_to_tagged_tokens2(tokens):
    # create a list of lists as [[token, tag], ...]
    tagged_tokens = []
    current_tag = const.NONE_TAG
    for token in tokens:
        if const.REPLACED_STARTING_TAGS in token:  # where tag begins: <comp.>
            splitted_token = token.split(const.REPLACED_STARTING_TAGS)
            current_tag = splitted_token[0]
            current_token = splitted_token[-1]
            if const.REPLACED_ENDING_TAGS in current_token:  # remove ending tag if it exists
                current_token = current_token.split(const.REPLACED_ENDING_TAGS)[0]
            current_token = current_token.strip()
            if current_token:
                tagged_tokens.append([current_token, current_tag])
            if const.REPLACED_ENDING_TAGS in token:
                current_tag = const.NONE_TAG
        elif const.REPLACED_ENDING_TAGS in token:  # where tag ends: </>
            token = token.split(const.REPLACED_ENDING_TAGS)[0]
            current_tag = const.NONE_TAG
            token = token.strip()
            if token:
                tagged_tokens.append([token, const.END_TAG])
            else:  # if the string is empty, then update the previous tag
                previous_tagged_token = tagged_tokens[-1]
                if previous_tagged_token[1] == const.IN_TAG:
                    previous_tagged_token[1] = const.END_TAG
                    tagged_tokens[-1] = previous_tagged_token
        elif current_tag != const.NONE_TAG:  # in the middle
            previous_tagged_token = tagged_tokens[-1]
            if previous_tagged_token[1] == const.NONE_TAG:  # it is actually the beginning
                tagged_tokens.append([token, current_tag])
            else:
                tagged_tokens.append([token, const.IN_TAG])
        else:
            tagged_tokens.append([token, const.NONE_TAG])

    return tagged_tokens
# convert tagged tokens to list of sentences, as required by packages such as gensim
# def tagged_tokens_to_sentences(tagged_tokens):
#


def tokens_from_file(file_path):
    text = ft.messy_codec_file_to_text(file_path).lower()
    tokens = ft.text_tokenizer(text)
    return tokens


def sentences_from_file(file_path):
    text = ft.messy_codec_file_to_text(file_path).lower()
    sentences = ft.text_to_sentences(text)
    return [ft.text_tokenizer(sentence) for sentence in sentences]


def tagged_tokens_from_file(file_path):
    text = ft.messy_codec_file_to_text(file_path).lower()
    text = replace_tag_marks_with_chars(text)
    tokens = ft.text_tokenizer(text)
    tagged_tokens = tokens_to_tagged_tokens2(tokens)
    # logging.info(file_path)
    return tagged_tokens


# returns a dict like {'comp': [[['esafetyworld', 'comp'], ['inc', 'end']]], 'date': [[['2000', 'date']],
# [['2001', 'date']]], 'item': [[['revenues', 'item']], [['profits', 'item']]]}
def entity_tagged_words_dict_from_tagged_tokens(tagged_tokens):
    current_tag = None
    entity_tagged_words_dict = {}
    for token, tag in tagged_tokens:
        if tag not in [const.NONE_TAG]:
            if tag not in [const.IN_TAG, const.END_TAG]:  # the beginning tag
                current_tag = tag
                if current_tag in entity_tagged_words_dict:
                    entity_tagged_words_dict[current_tag].append([[token,tag]])
                else:
                    entity_tagged_words_dict[current_tag] = [[[token,tag]]]
            else:
                entity_tagged_words_dict[current_tag][-1].append([token,tag])
    logging.info(entity_tagged_words_dict)
    return entity_tagged_words_dict


# returns a dict like: {'comp': [['esafetyworld', 'inc']], 'date': [['2000'], ['2001']], 'item': [['revenues'],
# ['profits']]}
def entity_dict_from_tagged_tokens(tagged_tokens):
    entity_dict = {}
    entity_tagged_words_dict = entity_tagged_words_dict_from_tagged_tokens(tagged_tokens)
    for entity, tagged_words in entity_tagged_words_dict.items():
        l = []
        for words in tagged_words:
            l.append(utils.sentence_from_tagged_ngram(words))
        entity_dict[entity] = l
    logging.info(entity_dict)
    return entity_dict


def ngrams_from_file(file_path, n, tagged=False):
    tokens = tagged_tokens_from_file(file_path) if tagged else tokens_from_file(file_path)
    ngrams = utils.ngram_from_list(tokens, n)
    return ngrams


def sequenced_ngrams_from_file(file_path, n, tagged=False):
    tokens = tagged_tokens_from_file(file_path) if tagged else tokens_from_file(file_path)
    ngrams = utils.sequenced_ngrams_from_list(tokens, n)
    return ngrams


def one_to_n_grams_from_file(file_path, n=5, tagged=False):
    grams = []
    for i in range(1, n + 1):
        sequenced_ngrams = sequenced_ngrams_from_file(file_path, i, tagged=tagged)
        # print('print(sequenced_ngrams)')
        # print(sequenced_ngrams)
        if tagged:
            sequenced_ngrams[:] = [tuple(utils.sentence_from_tagged_ngram(u)) for u in sequenced_ngrams]
        # sequenced_ngrams = utils.sentence_from_tagged_ngram(sequenced_ngrams) if tagged else sequenced_ngrams
        grams.append(sequenced_ngrams)
    return grams


def str_1_to_n_grams_from_file(file_path, n=5, tagged=False):
    grams = []
    for i in range(1, n + 1):
        sequenced_ngrams = sequenced_ngrams_from_file(file_path, i, tagged=tagged)
        # print('print(sequenced_ngrams)')
        # print(sequenced_ngrams)
        if tagged:
            sequenced_ngrams[:] = [tuple(utils.sentence_from_tagged_ngram(u)) for u in sequenced_ngrams]
        # sequenced_ngrams = utils.sentence_from_tagged_ngram(sequenced_ngrams) if tagged else sequenced_ngrams
        grams.append([utils.iter_to_string(tu) for tu in sequenced_ngrams])
    return grams


def tokens_from_dir(dir_path):
    tokens = []
    for path in ft.list_file_paths_under_dir(dir_path, ["txt"]):
        tokens += tokens_from_file(path)
    return tokens


def sentences_from_dir(dir_path):
    sentences = []
    for path in ft.list_file_paths_under_dir(dir_path, ["txt"]):
        sentences += sentences_from_file(path)
    return sentences


# for gram in ngrams_from_file(const.DATA_PATH + "examples/" + "34-53330.txt", 5, tagged=True):
#     print(gram)
# utils.display_logging_info()
# for path in ft.list_file_paths_under_dir(const.DATA_PATH + "examples/", ["txt"]):
#     tagged_tokens = tagged_tokens_from_file(path)
#     entity_dict = entity_dict_from_tagged_tokens(tagged_tokens)

# tagged_tokens = tagged_tokens_from_file(const.DATA_PATH + "examples/" + "34-53330.txt")
