from text_cleaning import aaer_corpus
import common.constants as const
from tensor2tensor.data_generators import generator_utils


def make_t2t_training_files():
    aaer = aaer_corpus.AAERParserNGrams(n=10)
    aaer.t2t_file_producer(target_size=5)


def make_t2t_vocal_file():
    aaer = aaer_corpus.AAERParserTokens()
    tokens = aaer.get_tokens()
    with open(const.T2T_TEMP_DIR + const.T2T_AAER_VOLCAB_NAME + '.40000', 'w') as f:
        f.write('\n'.join(tokens))

def make_vocal_file():
    aaer = aaer_corpus.AAERParserTokens()
    token_vocab = generator_utils.get_or_generate_vocab_inner(data_dir=const.T2T_DATA_DIR,
                                                              vocab_filename=const.T2T_AAER_VOLCAB_NAME,
                                                              vocab_size=40000,
                                                              generator=aaer.get_tokens())


# make_t2t_volcab_file()
# make_t2t_training_files()
make_vocal_file()
