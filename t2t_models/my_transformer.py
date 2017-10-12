import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor.models import transformer


@registry.register_model
class MyTransformer(transformer.Transformer):
    def decode(
            self,
            decoder_input,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams,
            cache=None):
        """Decode Transformer outputs from encoder representation.

        Args:
          decoder_input: inputs to bottom of the model.
              [batch_size, decoder_length, hidden_dim]
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encoder-decoder attention. [batch_size, input_length]
          decoder_self_attention_bias: Bias and mask weights for decoder
              self-attention. [batch_size, decoder_length]
          hparams: hyperparmeters for model.
          cache: dict, containing tensors which are the results of previous
              attentions, used for fast decoding.

        Returns:
          Final decoder representaiton. [batch_size, decoder_length, hidden_dim]
        """
        decoder_input = tf.nn.dropout(decoder_input,
                                      1.0 - hparams.layer_prepostprocess_dropout)

        decoder_output = transformer.transformer_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            hparams,
            cache=cache)
        # tf.Variable(decoder_output, name='decoder_output', validate_shape=False)
        tf.logging.info('decoder_output:')
        tf.logging.info(decoder_output)

        # Expand since t2t expects 4d tensors.
        return tf.expand_dims(decoder_output, axis=2)
