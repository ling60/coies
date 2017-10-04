import tensorflow as tf
import numpy as np
import text_cleaning.aaer_corpus as aaer
import text_cleaning.example_parsing as ex_parse
import common.utilities as util
import model_testing.tf_models as tf_models
import logging
import root_path
import common.constants as const


# wv.index2word is a list of vocab stored in wv, which could be used for mapping embeddings to words.
def tf_embeddings_from_aaer(aaer_model=None):
    aaer_model = aaer.AAERParserSentences() if aaer_model is None else aaer_model
    word2vec_model = aaer_model.make_word2vec_model()
    embedding_matrix = np.zeros((len(word2vec_model.wv.vocab), word2vec_model.layer1_size))
    for i in range(len(word2vec_model.wv.vocab)):
        embedding_vector = word2vec_model.wv[word2vec_model.wv.index2word[i]]
        embedding_matrix[i] = embedding_vector
    # tf_embeddings = tf.constant(embedding_matrix)
    return embedding_matrix, word2vec_model.wv.index2word


# returns a similarity matrix based on input embeddings given by word indexes
def tf_cosine_similarity(embedding, word_indexes):
    # create the cosine similarity operations
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embeddings = tf.divide(embedding, norm)
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, word_indexes)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    return similarity


# two methods to do this:
# 1.context = (window1) + target_term + (window2), then predict/generate the whole context
# 2:context = sub_context + target_term + (window) LSTM could be used.
# 3.given: context + (window) + target_term, try to predict/generate window

class PTBInput(object):
    """The input data."""
    def __init__(self, config, data, name=None, is_testing=False, window_size=2):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        logging.info('PTBInput:num_steps:%d', self.num_steps)
        self.epoch_size = self.compute_epoch_size(len(data), self.batch_size, window_size, self.num_steps, is_testing)
        print('epoch_size:', self.epoch_size)
        self.window_size = window_size
        self.input_data = None
        self.targets = None

        self.get_data(data, name=name, is_testing=is_testing)
        # print(self.targets, self.input_data)

    def get_data(self, data, name, is_testing):
        self.input_data, self.targets = self.data_producer(
            data, self.batch_size, self.num_steps, name=name, is_testing=is_testing, window_size=self.window_size)

    @staticmethod
    def compute_epoch_size(data_len, batch_size, window_size, num_steps,  is_testing):
        # if is_testing:
        #     epoch_size = (data_len // batch_size) - window_size - num_steps
        # else:
        #     epoch_size = ((data_len // batch_size) - window_size) // num_steps

        epoch_size = (data_len // batch_size) - window_size - num_steps
        return epoch_size

    @staticmethod
    def data_producer(raw_data, batch_size, num_steps, is_testing=False, window_size=2, name=None):
        """Iterate on the raw AAER data.
      Code from: https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py:ptb_producer
      This chunks up raw_data into batches of examples and returns Tensors that
      are drawn from these batches.
      Args:
        raw_data: one of the raw data outputs from ptb_raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls.
        name: the name of this operation (optional).
        is_testing: flag if the data is for testing.If true, the output will be ngrams like, instead of windows.
        window_size: time shifting distance between y and x
      Returns:
        A pair of Tensors, each shaped [batch_size, num_steps]. The second element
        of the tuple is the same data time-shifted to the right by window size.
      Raises:
        tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
      """
        with tf.name_scope(name, "DataProducer", [raw_data, batch_size, num_steps]):
            logging.info("batch_len")
            logging.info(len(raw_data) // batch_size)
            logging.info('num_steps: %d', num_steps)
            raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

            data_len = tf.size(raw_data)
            batch_len = data_len // batch_size
            data = tf.reshape(raw_data[0: batch_size * batch_len],
                              [batch_size, batch_len])

            epoch_size = PTBInput.compute_epoch_size(data_len, batch_size, window_size, num_steps, is_testing)

            assertion = tf.assert_positive(
                epoch_size,
                message="epoch_size == 0, decrease batch_size or num_steps")
            with tf.control_dependencies([assertion]):
                epoch_size = tf.identity(epoch_size, name="epoch_size")

            i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
            # if is_testing:
            #     slice_pos = i
            # else:
            #     slice_pos = i * num_steps

            slice_pos = i

            x = tf.strided_slice(data, [0, slice_pos],
                                 [batch_size, slice_pos + num_steps])
            x.set_shape([batch_size, num_steps])
            y = tf.strided_slice(data, [0, slice_pos + window_size],
                                 [batch_size, slice_pos + num_steps + window_size])
            y.set_shape([batch_size, num_steps])
            return x, y


class TestInput(PTBInput):
    # with replace_options set, this class will provide input(x) data where ending replaced by words given. Targets(y)
    # will not be changed
    def __init__(self, config, data, window_size=2, replace_options=None, name=None):
        # replace_options: {skip_n: number of words to be replaced; words: new words introduced to replace}
        self.replace_options = replace_options
        self.replaced_data = None  # the original data being replaced, which is what we are looking for.
        super().__init__(config, data, is_testing=True, window_size=window_size, name=name)

    def get_data(self, data, name, is_testing):
        super().get_data(data, name, is_testing)

        if self.replace_options is not None:
            num_skip = self.replace_options['skip_n']
            words = self.replace_options['words']
            assert type(words) is list
            words = np.array(words)
            num_steps_left = self.num_steps - num_skip
            x_left = tf.strided_slice(self.input_data, [0, 0], [self.batch_size, num_steps_left])
            self.replaced_data = tf.strided_slice(self.input_data, [0, num_steps_left],
                                                  [self.batch_size, self.num_steps])
            x_left.set_shape([self.batch_size, num_steps_left])
            words = np.resize(words, (self.batch_size, len(words)))
            x_right = tf.constant(words, dtype=tf.int32)
            self.input_data = tf.concat([x_left, x_right], 1)


def method_2():
    window_size = 2
    sub_context_size = 10
    target_size = 1
    context_size = sub_context_size + target_size + window_size  # number of steps for rnn
    batch_size = 20

    # training data
    model = aaer.AAERParserTokens()
    tokens = model.get_tokens()
    embeddings, index2word = tf_embeddings_from_aaer()
    word2index = {word: i for i, word in enumerate(index2word)}
    tokens_as_indexes = [word2index[i] for i in tokens]

    # test data
    test_tokens = ex_parse.tagged_tokens_from_file(const.EXAMPLE_FILE)
    test_indexes = [word2index[t[0]] for t in test_tokens]
    test_word = 'profits'

    config = tf_models.SmallConfig()
    config.num_steps=10
    eval_config = tf_models.SmallConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 5
    test_only = True

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=tokens_as_indexes, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = tf_models.PTBModel(is_training=True, config=config,
                                       input_=train_input, embedding_matrix=embeddings)
            tf.summary.scalar("Training_Loss", m.cost)
            tf.summary.scalar("Learning_Rate", m.lr)

        with tf.name_scope("Test"):
            test_input = TestInput(config=eval_config, data=test_indexes, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = tf_models.PTBModel(is_training=False, config=eval_config,
                                           input_=test_input, embedding_matrix=embeddings)

        with tf.name_scope("Find"):
            test_input = TestInput(config=eval_config, data=test_indexes, name="FindInput",
                                   replace_options={'skip_n':1, 'words': [word2index[test_word]]})
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mfind = tf_models.PTBModel(is_training=False, config=eval_config,
                                           input_=test_input, embedding_matrix=embeddings)
        # metagraph = tf.train.export_meta_graph()
        #
        # tf.train.import_meta_graph(metagraph)
        save_path = root_path.GENERATED_DATA_DIR
        sv = tf.train.Supervisor(logdir=save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=False)
        with sv.managed_session(config=config_proto) as session:
            if not test_only:
                for i in range(config.max_max_epoch):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                    m.assign_lr(session, config.learning_rate * lr_decay)

                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                    train_perplexity, _= tf_models.run_epoch(session, m, eval_op=m.train_op,
                                                 verbose=True)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                    # valid_perplexity = tf_models.run_epoch(session, mvalid)
                    # print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity, details= tf_models.run_epoch(session, mtest, is_testing=True)
            test_losses, test_inputs = details[:2]
            print("Test Perplexity: %.3f" % test_perplexity)
            top_inputs = [test_inputs[l].tolist()[0] for l in util.top_n_from_list(test_losses, 3, start_max=False)]
            print(top_inputs)
            print([[index2word[i] for i in l] for l in top_inputs])

            find_perplexity, details = tf_models.run_epoch(session, mfind, is_testing=True, is_finding=True)
            find_losses, find_inputs, replaced_words = details
            print("Find/replaced Perplexity: %.3f" % find_perplexity)

            np_t_losses = np.array(test_losses)
            np_f_losses = np.array(find_losses)
            change_rate = abs((np_t_losses-np_f_losses)/np_t_losses)

            # print('replaced words:', replaced_words)
            for i in util.top_n_from_list(change_rate.tolist(), 3, start_max=False):
                indexes = find_inputs[i].tolist()[0]
                contexts = [index2word[n] for n in indexes]
                replaced_indexes = replaced_words[i].tolist()[0]
                replaced = [index2word[n] for n in replaced_indexes]

                print(indexes)
                print(contexts)
                print(replaced)

            if save_path:
                print("Saving model to %s." % save_path)
                sv.saver.save(session, save_path, global_step=sv.global_step)
            # tokens_as_embeddings = tf.nn.embedding_lookup(embeddings, tokens_as_indexes)
            # print([tokens_as_embeddings])
            # for ngrams in tokens:
            #     ngrams_as_indexes = []
            #     for word in ngrams:
            #         # print(word)
            #         ngrams_as_indexes.append(word2index[word])
            #     tokens_as_indexes.append(ngrams_as_indexes)


def method_3():
    window_size = 2
    context_size = 5
    target_size = 1
    gram_size = window_size + context_size + target_size

    model = aaer.AAERParserNGrams(n=gram_size)
    tokens = model.get_tokens()
    embeddings, index2word = tf_embeddings_from_aaer()
    word2index = {word: i for i, word in enumerate(index2word)}
    tokens_as_indexes = []
    for ngrams in tokens:
        ngrams_as_indexes = []
        for word in ngrams:
            # print(word)
            ngrams_as_indexes.append(word2index[word])
        tokens_as_indexes.append(ngrams_as_indexes)
    for i_grams in tokens_as_indexes:
        print([index2word[i] for i in i_grams[0:context_size]])
        print([index2word[i] for i in i_grams[context_size:context_size + window_size]])
        # tokens_as_indexes = [[word2index[word] for word in ngram] for ngram in tokens]


util.display_logging_info()
method_2()


# valid_size = 16  # Random set of words to evaluate similarity on.
# valid_window = 100  # Only pick dev samples in the head of the distribution.
# valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
#
# # embedding layer weights are frozen to avoid updating embeddings while training
# saved_embeddings, indexed_words = tf_embeddings_from_aaer()
# embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)
#
# # create the cosine similarity operations
# norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
# normalized_embeddings = embedding / norm
# valid_embeddings = tf.nn.embedding_lookup(
#       normalized_embeddings, valid_dataset)
# similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Add variable initializer.
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     # call our similarity operation
#     sim = tf_cosine_similarity(embedding, valid_dataset).eval()
#     # sim = similarity.eval()
#     # run through each valid example, finding closest words
#     for i in range(valid_size):
#         valid_word = indexed_words[i]
#         top_k = 8  # number of nearest neighbors
#         nearest = (-sim[i, :]).argsort()[1:top_k + 1]
#         log_str = 'Nearest to %s:' % valid_word
#         for k in range(top_k):
#             close_word = indexed_words[nearest[k]]
#             log_str = '%s %s,' % (log_str, close_word)
#         print(log_str)
