from t2t_models import text_encoding
import os
import tensorflow as tf
import text_cleaning.example_parsing as ex_parsing
import common.constants as const
import common.file_tools as ft
import model_testing.dl_context_models as dl_context

N_GRAMS = 10
test_file_source = ft.get_source_file_by_example_file(const.TEST_FILE)
tokens = ex_parsing.ngrams_from_file(test_file_source, N_GRAMS, tagged=False)
# eval_tokens = []
# for t in tokens:
#     s = t[:-1] + 'profits'.split(' ')
#     eval_tokens.append(s)
# # TRAIN_DIR=$DATA_DIR/train/$PROBLEM/$MODEL-$HPARAMS
# print(eval_tokens)
# t = text_encoding.TextEncoding(tokens, eval_tokens)
# t.encode()

m_t2t = dl_context.T2TContextModel(load_aaer_data=True)
# m_t2t._make_docvec_dict(tokens)
print(m_t2t._docvec_dict)
