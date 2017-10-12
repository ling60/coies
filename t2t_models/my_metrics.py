from tensor2tensor.utils import metrics

import tensorflow as tf


class MyMetrics(metrics.Metrics):
    pass


def padded_accuracy_top5(predictions,
                         labels):
    return predictions
