import os
import sys
import csv
import common.file_tools as file_tools
import common.constants as const


def list_files_by_search_string(path, search_str):
    file_count = 0
    for root, dirs, files in os.walk(path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            text = open(file_path, 'r', encoding='utf-8').read()
            if search_str in text and 'quarter'  in text:
                file_count += 1
                print(file_path)
    print(file_count)


# print(sys.getdefaultencoding())
# list_files_by_search_string(DATA_PATH + "sec/aaer", "financial statement")

# return start, end offset
def decode_offsets(offset_string):
    assert type(offset_string) == str
    offsets = offset_string.split('_')
    return int(offsets[0]), int(offsets[1])


# due to different offsets counting method between python and notepad++, we convert the notepad++ offsets to fit
# python text file reading
def string_from_offsets(text, int_start, int_end):
    lines = text.splitlines()


# return company name, date of financial report forgery, and the item effected
def parse_company_date_item(index_file_path, text_dir_path):
    text_dir_path = file_tools.check_dir_ending(text_dir_path)
    with open(index_file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            print(row)
            text_file_path = os.path.join(text_dir_path, row[0] + '.txt')
            with open(text_file_path, 'r', encoding='utf-8') as textfile:
                text = textfile.read()
                lines = text.splitlines()
                print(lines[24][92:])
                # print(len(text), sum([len(line)+2 for line in lines]))
                company_name_offsets = decode_offsets(row[1])
                # print(text[company_name_offsets[0]:company_name_offsets[1]])


# parse_company_date_item(const.DATA_PATH + 'examples/fileno-company-date-item.txt', const.DATA_PATH + const.AAER_PATH)