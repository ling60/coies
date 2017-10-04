import fasttext
import common.constants as const
import os

file_path = os.path.join(const.TEST_DIR, "34-73931.txt")
# print(file_path)
# print(open(file_path).read())
model = fasttext.skipgram(file_path, 'fasttext_model', encoding='utf-8', min_count=1)

print(model.words)