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

    @staticmethod
    def get_tensor():
        return tf.get_default_graph().get_tensor_by_name(
            'body/model/parallel_0/body/decoder/layer_5/ffn/layer_postprocess/layer_norm/add_1:0')

    def before_run(self, run_context):
        t = self.get_tensor()
        return SessionRunArgs(t)

    def after_run(self, run_context, run_values):
        embedding_output = run_values.results
        # print(run_values.results)
        self.embeddings.append(embedding_output)

    def end(self, session):
        pass


class LossHook(EmbeddingsHook):
    @staticmethod
    def get_tensor():
        return tf.get_default_graph().get_tensor_by_name(
            'losses_avg/problem_0/total_loss:0')
