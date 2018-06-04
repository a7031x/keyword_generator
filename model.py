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
        if self.train_mode:
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
            self.input_target_question = tf.placeholder(tf.int32, shape=[None, None], name='target_question')
            self.question_mask, self.question_len = func.tensor_to_mask(self.input_target_question)
            self.max_question_len = tf.reduce_max(self.question_len)


    def feed(self, aids, qids=None, qv=None, st=None, keep_prob=1.0):
        feed_dict = {
            self.input_word: aids,
            self.input_keep_prob: keep_prob
        }
        if qv is not None:
            for v in qv:
                v[config.EOS_ID] = 1
            feed_dict[self.input_label_question_vector] = qv
        if st is not None:
            feed_dict[self.input_label_answer] = st
        if qids is not None:
            feed_dict[self.input_label_question] = qids
            tids = [[id for id in x if id != config.NULL_ID] + [config.EOS_ID] for x in qids]
            mlen = len(qids[0]) + 1
            tids = [x + [config.NULL_ID] * (mlen-len(x)) for x in tids]
            feed_dict[self.input_target_question] = tids
        return feed_dict


    def create_embeddings(self):
        with tf.name_scope('embedding'):
            self.answer_embedding = tf.get_variable(name='answer_embedding', shape=[self.answer_vocab_size, config.embedding_dim])
            self.question_embedding = tf.get_variable(name='question_embedding', shape=[self.question_vocab_size, config.embedding_dim])
            self.emb = tf.nn.embedding_lookup(self.answer_embedding, self.input_word, name='emb')
            tf.summary.histogram('embedding/question_embedding', self.question_embedding)
            tf.summary.histogram('embedding/answer_embedding', self.answer_embedding)
            tf.summary.histogram('embedding/emb', self.emb)


    def create_encoder(self):
        with tf.name_scope('encoder'):
            fw_cell = rnn.create_rnn_cell('lstm', config.encoder_hidden_dim, config.num_encoder_rnn_layers, config.num_encoder_residual_layers, self.input_keep_prob)
            bw_cell = rnn.create_rnn_cell('lstm', config.encoder_hidden_dim, config.num_encoder_rnn_layers, config.num_encoder_residual_layers, self.input_keep_prob)
            '''
            fw_cell = []
            bw_cell = []
            for i in range(config.num_encoder_rnn_layers):
                fw = tf.contrib.rnn.BasicLSTMCell(config.encoder_hidden_dim)
                bw = tf.contrib.rnn.BasicLSTMCell(config.encoder_hidden_dim)
                fw = tf.contrib.rnn.DropoutWrapper(cell=fw, input_keep_prob=self.input_keep_prob)
                bw = tf.contrib.rnn.DropoutWrapper(cell=bw, input_keep_prob=self.input_keep_prob)
                if i == 0:
                    fw = tf.contrib.rnn.ResidualWrapper(fw, residual_fn=None)
                    bw = tf.contrib.rnn.ResidualWrapper(bw, residual_fn=None)
                fw_cell.append(fw)
                bw_cell.append(bw)
            fw_cell = tf.contrib.rnn.MultiRNNCell(fw_cell)
            bw_cell = tf.contrib.rnn.MultiRNNCell(bw_cell)
            '''
            bi_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.emb, dtype=tf.float32, sequence_length=self.length, swap_memory=True)
            self.encoder_output = tf.concat(bi_output, -1)
            encoder_state = []
            for layer_id in range(config.num_encoder_rnn_layers):
                encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                encoder_state.append(bi_encoder_state[1][layer_id])  # backward
            self.encoder_state = tuple(encoder_state)
            tf.summary.histogram('encoder/encoder_output', self.encoder_output)
            tf.summary.histogram('encoder/encoder_state', self.encoder_state)


    def create_decoder(self):
        with tf.variable_scope('decoder/output_projection'):
            output_layer = layers_core.Dense(self.question_vocab_size, use_bias=False, name='output_projection')
        with tf.name_scope('decoder'), tf.variable_scope('decoder') as decoder_scope:
            if self.train_mode:
                memory = self.encoder_output
                source_sequence_length = self.length
                encoder_state = self.encoder_state
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
            start_tokens = tf.fill([self.batch_size], config.SOS_ID)
            if self.train_mode:
                target_input = tf.concat([tf.expand_dims(start_tokens, -1), self.input_label_question], 1)
                decoder_emb = tf.nn.embedding_lookup(self.question_embedding, target_input)
                helper = ts.TrainingHelper(decoder_emb, self.question_len)
                decoder = ts.BasicDecoder(cell, helper, decoder_initial_state)
                output, self.final_context_state, _ = ts.dynamic_decode(decoder, swap_memory=True, scope=decoder_scope)
                self.question_logit = output_layer(output.rnn_output)
                tf.summary.histogram('decoder/logit', self.question_logit)
            else:
                decoder = ts.BeamSearchDecoder(
                    cell=cell,
                    embedding=self.question_embedding,
                    start_tokens=start_tokens,
                    end_token=config.EOS_ID,
                    initial_state=decoder_initial_state,
                    beam_width=config.beam_width,
                    output_layer=output_layer,
                    length_penalty_weight=0.0)
                output, self.final_context_state, _ = ts.dynamic_decode(decoder, config.max_question_len, scope=decoder_scope)
                self.sample_id = output.predicted_ids


    def calc_regularization_loss(self):
        loss = 0
        for variable in tf.trainable_variables():
            loss = tf.nn.l2_loss(variable) + loss
        return loss * 0.0001


    def create_seq_loss(self):
        with tf.name_scope('seq_loss'):
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_target_question, logits=self.question_logit) * self.question_mask
            self.regularization_loss = self.calc_regularization_loss()
            self.seq_loss = tf.reduce_sum(crossent) / tf.to_float(self.batch_size)
            tf.summary.scalar('seq_loss', self.seq_loss)

    
    def create_vector_loss(self):
        with tf.name_scope('vector_loss'):
            mask = tf.expand_dims(self.question_mask, -1)
            hardmax = ts.hardmax(self.question_logit)
            self.max_logit = hardmax * self.question_logit * mask
            self.squeezed_logit = tf.reduce_sum(self.max_logit, 1)
            vector_weight = tf.cast(tf.reduce_sum(hardmax, 1) + self.input_label_question_vector > 0.0, tf.float32) * self.question_word_weight
            vector_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label_question_vector, logits=self.squeezed_logit) * vector_weight
            self.vector_loss = tf.reduce_mean(tf.reduce_sum(vector_loss, -1))
            tf.summary.scalar('vector_loss', self.vector_loss)


    def create_loss(self):
        self.create_vector_loss()
        self.create_seq_loss()
        with tf.name_scope('loss'):
            self.loss = self.vector_loss + self.seq_loss
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
