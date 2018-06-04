import tensorflow as tf
import tensorflow.contrib.seq2seq as ts
from tensorflow.python.layers import core as layers_core
import rnn_helper as rnn
import numpy as np
import utils
import func
import config
import os

class Model(object):
    def __init__(self, qi2c, ckpt_folder, train_mode=True, name='model'):
        self.name = name
        self.ckpt_folder = ckpt_folder
        self.question_vocab_size = config.question_vocab_size
        self.answer_vocab_size = config.answer_vocab_size
        self.train_mode = train_mode
        qww = [0] * self.question_vocab_size
        pmax = qi2c[80]
        for i,c in qi2c.items():
            qww[i] = min(c, pmax)
        qww = np.array(qww) ** 0.5
        self.question_word_weight = 5 - 4.7 * qww / np.max(qww)
        print(self.question_word_weight[:50])
        if self.ckpt_folder is not None:
            utils.mkdir(self.ckpt_folder)
        initializer = tf.random_uniform_initializer(-0.05, 0.05)
        with tf.variable_scope(self.name, initializer=initializer):
            self.initialize()


    def initialize(self):
        self.create_inputs()
        self.create_embeddings()
        self.create_encoder()
        self.create_decoder()
        self.create_loss()
        self.create_optimizer()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        total, vc = self.number_parameters()
        print('trainable parameters: {}'.format(total))
        for name, count in vc.items():
            print('{}: {}'.format(name, count))


    def create_inputs(self):
        with tf.name_scope('input'):
            self.input_word = tf.placeholder(tf.int32, shape=[None, None], name='word')
            self.input_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.batch_size = tf.shape(self.input_word)[0]
            self.mask, self.length = func.tensor_to_mask(self.input_word)
            self.input_label_answer = tf.placeholder(tf.float32, shape=[None, None], name='label_answer')
            self.input_label_question = tf.placeholder(tf.int32, shape=[None, None], name='label_question')
            self.input_label_question_vector = tf.placeholder(tf.float32, shape=[None, self.question_vocab_size], name='label_question_vector')
            self.question_mask, self.question_len = func.tensor_to_mask(self.input_label_question)
            self.max_question_len = tf.reduce_max(self.question_len)


    def feed(self, aids, qids=None, qv=None, st=None, keep_prob=1.0):
        feed_dict = {
            self.input_word: aids,
            self.input_keep_prob: keep_prob
        }
        if qv is not None:
            feed_dict[self.input_label_question_vector] = qv
        if st is not None:
            feed_dict[self.input_label_answer] = st
        if qids is not None:
            feed_dict[self.input_label_question] = qids
        return feed_dict


    def create_embeddings(self):
        with tf.name_scope('embedding'):
            self.question_embedding = tf.get_variable(name='question_embedding', shape=[self.question_vocab_size, config.embedding_dim])
            self.answer_embedding = tf.get_variable(name='answer_embedding', shape=[self.answer_vocab_size, config.embedding_dim])
            self.emb = tf.nn.embedding_lookup(self.answer_embedding, self.input_word, name='emb')
            tf.summary.histogram('embedding/question_embedding', self.question_embedding)
            tf.summary.histogram('embedding/answer_embedding', self.answer_embedding)
            tf.summary.histogram('embedding/emb', self.emb)


    def create_encoder(self):
        with tf.name_scope('encoder'):
            fw_cell = rnn.create_rnn_cell('lstm', config.encoder_hidden_dim, config.num_encoder_rnn_layers, config.num_encoder_residual_layers, self.input_keep_prob)
            bw_cell = rnn.create_rnn_cell('lstm', config.encoder_hidden_dim, config.num_encoder_rnn_layers, config.num_encoder_residual_layers, self.input_keep_prob)
            bi_output, self.encoder_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.emb, sequence_length=self.length)
            self.encoder_output = tf.concat(bi_output, -1)
            tf.summary.histogram('encoder/encoder_output', self.encoder_output)
            tf.summary.histogram('encoder/encoder_state', self.encoder_state)


    def create_decoder(self):
        with tf.name_scope('decoder'):
            output_layer = layers_core.Dense(self.question_vocab_size, use_bias=False, name='output_projection')
            if self.train_mode:
                batch_size = self.batch_size
            else:
                memory = tf.contrib.seq2seq.tile_batch(self.encoder_output, multiplier=config.beam_width)
                source_sequence_length = ts.tile_batch(self.length, multiplier=config.beam_width)
                encoder_state = tf.contrib.seq2seq.tile_batch(self.encoder_state, multiplier=config.beam_width)
                batch_size = self.batch_size * config.beam_width

            attention_mechanism = ts.LuongAttention(config.decoder_hidden_dim, memory, source_sequence_length, scale=True)
            cell = rnn.create_rnn_cell('lstm', config.decoder_hidden_dim, config.num_decoder_rnn_layers, config.num_decoder_residual_layers, self.input_keep_prob)
            cell = ts.AttentionWrapper(cell, attention_mechanism,
                attention_layer_size=config.decoder_hidden_dim,
                alignment_history=(not self.train_mode) and (config.beam_width == 0),
                output_attention=True,
                name='attention')

            decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
            if self.train_mode:
                decoder_emb = tf.nn.embedding_lookup(self.question_embedding, self.input_label_question)
                helper = ts.TrainingHelper(decoder_emb, self.question_len)
                decoder = ts.BasicDecoder(cell, helper, decoder_initial_state)
                output, self.final_context_state, _ = ts.dynamic_decode(decoder, scope='decoder')
                self.question_logit = output_layer(output.rnn_output)
                tf.summary.histogram('decoder/logit', self.question_logit)
                tf.summary.histogram('decoder/context_state', self.final_context_state)
            else:
                start_tokens = tf.fill([self.batch_size], config.SOS_ID)
                decoder = ts.BeamSearchDecoder(
                    cell=cell,
                    embedding=self.question_embedding,
                    start_tokens=start_tokens,
                    end_token=config.EOS_ID,
                    initial_state=decoder_initial_state,
                    beam_width=config.beam_width,
                    output_layer=output_layer,
                    length_penalty_weight=0.0)
                output, self.final_context_state, _ = ts.dynamic_decode(decoder, config.max_question_len, scope='decoder')
                self.sample_id = output.predicted_ids


    def calc_regularization_loss(self):
        loss = 0
        for variable in tf.trainable_variables():
            loss = tf.nn.l2_loss(variable) + loss
        return loss * 0.0001


    def create_loss(self):
        with tf.name_scope('loss'):
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label_question, logits=self.question_logit) * self.question_mask
            self.regularization_loss = self.calc_regularization_loss()
            self.loss = tf.reduce_sum(crossent) / tf.to_float(self.batch_size)
            tf.summary.scalar('regularization_loss', self.regularization_loss)
            tf.summary.scalar('loss', self.loss)


    def create_optimizer(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.opt = tf.train.AdamOptimizer(learning_rate=1E-3)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)


    def restore(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_folder)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('MODEL LOADED.')
        else:
            sess.run(tf.global_variables_initializer())


    def save(self, sess):
        self.saver.save(sess, os.path.join(self.ckpt_folder, 'model.ckpt'))


    def summarize(self, writer):
        self.summary = tf.summary.merge_all()


    def number_parameters(self):
        total_parameters = 0
        vc = {}
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
            vc[variable.name] = variable_parameters
        return total_parameters, vc


if __name__ == '__main__':
    from data import Dataset
    data = Dataset()
    model = Model(data.qi2c, None)
