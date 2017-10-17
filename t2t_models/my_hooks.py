import tensorflow as tf

from tensorflow.python.training.session_run_hook import SessionRunArgs


def log_tf_var_with_name(var, var_name, session):
    if var.name == var_name:
        print("var name:%s" % var.name)
        print(session.run(var))


class EmbeddingsHook(tf.train.SessionRunHook):
    def __init__(self):
        self.embeddings = []

    def begin(self):
        # self.t_embeddings = tf.concat(self.embeddings, 0)
        pass

    def before_run(self, run_context):
        t = tf.get_default_graph().get_tensor_by_name(
            'body/model/parallel_0/body/decoder/layer_5/ffn/layer_postprocess/layer_norm/add_1:0')
        return SessionRunArgs(t)
        # t_output = run_context.session.run(t)
        # print('before_run:')
        # print(t_output.shape)
        # print(t_output)
        # t_output = run_context.session.run(t)
        # print('before_run:')
        # print(t_output)

    def after_run(self, run_context, run_values):
        embedding_output = run_values.results
        # print(run_values.results)
        self.embeddings.append(embedding_output)

        # super().after_run(run_context, run_values)
        # with open(os.path.join(const.GENERATED_DATA_DIR, 't2t_variables_after_run.txt'), 'w') as f:
        #     for v in tf.get_default_graph().get_operations():
        #         print(v)
        #         f.write('%s\n' % v)
        # raise MemoryError
        # t = tf.get_default_graph().get_tensor_by_name(
        #     'body/model/parallel_0/body/decoder/layer_5/ffn/layer_postprocess/layer_norm/add_1:0')
        # t_output = run_context.session.run(t)
        # t = tf.get_default_graph().get_tensor_by_name(
        #     'body/model/parallel_0/body/decoder/layer_5/ffn/layer_postprocess/layer_norm/add_1:0')
        # t_output = run_context.session.run(t)

        # for var in tf.global_variables():
        #     tf.get_default_graph().get_tensor_by_name('')
        #     log_tf_var_with_name(var, 'body/model/parallel_0/body/decoder/layer_5/ffn/layer_postprocess/layer_norm/add_1:0', run_context.session)
        #     log_tf_var_with_name(var, 'losses_avg/problem_0/training_loss:0', run_context.session)
        #     log_tf_var_with_name(var, 'losses_avg/problem_0/total_loss:0', run_context.session)

    def end(self, session):

        pass
