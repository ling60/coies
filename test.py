import ner

import logging
import json
import common.constants as const
import os

logging.basicConfig(level=logging.INFO)
path = const.EXAMPLE_FILE
tagger = ner.SocketNER(host='localhost', port=8081)
with open(path) as f:
    s = f.read()
    e_dict = tagger.get_entities(s)
    print(e_dict)
    print(e_dict.keys())

# with open(path) as f:
#     s = f.read()
#     s = s.replace("\'", "\"")
#     adict = json.loads(s)
#     print(adict)
#     print(*adict.values())
#     print(sum(adict.values()))

# a = cb.PhraseVecBigrams()
# print(a.aaer_model.get_bigrams(['esafetyworld', 'inc']))
# a = t2t_encoding.TextEncoding(['esafetyworld', 'inc'])
# a.encode()
