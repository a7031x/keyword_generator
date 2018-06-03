import tensorflow as tf
import numpy as np
import utils
import func
import config
import os

class Model(object):
    def __init__(self, qi2c, ckpt_folder, name='model'):
        self.name = name
        self.ckpt_folder = ckpt_folder
        self.question_vocab_size = config.question_vocab_size
        self.answer_vocab_size = config.answer_vocab_size
        qww = [0] * self.question_vocab_size
        pmax = qi2c[16]
        for i,c in qi2c.items():
            qww[i] = min(c, pmax)
        qww = np.array(qww) ** 0.5
        self.question_word_weight = 5 - 4 * qww / np.max(qww)
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
        self.create_self_match()
        self.create_decoder()
        self.create_seq_decoder()
        self.create_attention()
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
            self.question_grid_len = tf.reduce_max(self.question_len)


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
            self.encoding, _ = func.rnn('bi-lstm', self.emb, self.length, config.encoder_hidden_dim, 4, self.input_keep_prob)
            self.encoding = tf.nn.dropout(self.encoding, self.input_keep_prob, name='encoding')
            b_fw = self.encoding[:,-1,:config.encoder_hidden_dim]
            b_bw = self.encoding[:,0,config.encoder_hidden_dim:]
            self.encoder_last_state = tf.concat([b_fw, b_bw], -1)
            tf.summary.histogram('encoder/encoding', self.encoding)
            tf.summary.histogram('encoder/last_state', self.encoder_last_state)


    def create_self_match(self):
        with tf.name_scope('self_match'), tf.variable_scope('self_match'):
            self_att = func.dot_attention(self.encoding, self.encoding, self.mask, config.encoder_hidden_dim, self.input_keep_prob)
            self.self_match, _ = func.rnn('gru', self_att, self.length, config.encoder_hidden_dim, 1, self.input_keep_prob)
            tf.summary.histogram('attention/self_match', self.self_match)


    def create_decoder(self):
        with tf.name_scope('decoder'):
            self.ws_answer = tf.get_variable(name='ws_answer', shape=[config.encoder_hidden_dim, 1])
            self.ws_question = tf.get_variable(name='ws_question', shape=[config.encoder_hidden_dim, self.question_vocab_size])
            self.encoding_sum = tf.reduce_sum(self.self_match, axis=1)
            self.answer_logit = tf.einsum('aij,jk->aik', self.self_match, self.ws_answer)
            self.answer_logit = tf.squeeze(self.answer_logit, [-1], name='answer_logit')
            self.question_logit = tf.clip_by_value(tf.matmul(self.encoding_sum, self.ws_question), -10, 10, name='question_logit')
            tf.summary.histogram('decoder/answer_logit', self.answer_logit)
            tf.summary.histogram('decoder/question_logit', self.question_logit)


    def create_seq_decoder(self):
        with tf.name_scope('seq_decoder'):
            init_state = func.dense(self.self_match, config.decoder_hidden_dim*2, scope='seq_decoder_init_state')
            init_state = tf.reduce_sum(init_state, 1)
            initial_state = tf.contrib.rnn.LSTMStateTuple(init_state[:,:config.decoder_hidden_dim], init_state[:,config.decoder_hidden_dim:])
            output_sequence, self.decoder_length = func.rnn_decode(
                'lstm', self.batch_size, config.decoder_hidden_dim, self.question_embedding,
                config.SOS_ID, config.EOS_ID, (initial_state,), self.question_grid_len)
            self.decoder_hidden = tf.identity(output_sequence.rnn_output, 'decoder_hidden')
            tf.summary.histogram('seq_decoder/hidden', self.decoder_hidden)


    def create_attention(self):
        with tf.name_scope('qa_attention'), tf.variable_scope('qa_attention'):
            self.ct = func.dot_attention(self.decoder_hidden, self.self_match, self.mask, config.dot_attention_dim, self.input_keep_prob)
            self.combined_h = tf.concat([self.decoder_hidden, self.ct], -1, name='combined_h')#[batch, question_len, 450]           
            self.wt = tf.get_variable('wt', shape=[config.max_question_len, self.combined_h.get_shape()[-1], config.decoder_hidden_dim])
            self.ws = tf.get_variable('ws', shape=[config.decoder_hidden_dim, self.question_vocab_size])
            self.decoder_question_len = tf.shape(self.combined_h)[1]
            self.wt_h = tf.einsum('bij,ijk->bik', self.combined_h, self.wt[:self.decoder_question_len,:,:], name='wt_h')
            self.ws_tanh_wt = tf.einsum('bik,kj->bij', tf.tanh(self.wt_h), self.ws)
            #self.question_sequence_logit = func.softmax(self.ws_tanh_wt, tf.expand_dims(self.question_mask, -1))
            self.question_sequence_logit = tf.clip_by_value(self.ws_tanh_wt, -10, 10, name='question_sequence_logit')            
            tf.summary.histogram('qa_attention/question_sequence_logit', self.question_sequence_logit)


    def create_loss(self):
        with tf.name_scope('loss'):
            self.answer_loss = func.cross_entropy(tf.sigmoid(self.answer_logit), self.input_label_answer, self.mask, pos_weight=3.0)
            self.question_vector_loss = func.cross_entropy(tf.sigmoid(self.question_logit), self.input_label_question_vector, None, pos_weight=self.question_word_weight)
            self.answer_loss = tf.reduce_mean(tf.reduce_sum(self.answer_loss, -1))
            self.question_vector_loss = tf.reduce_mean(tf.reduce_sum(self.question_vector_loss, -1))

            self.final_question_mask = tf.sequence_mask(self.question_grid_len, dtype=tf.float32)
            target = tf.one_hot(self.input_label_question, self.question_vocab_size, dtype=tf.float32)
            self.question_sequence_loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(
                logits=self.question_sequence_logit[:,:self.question_grid_len,:],
                targets=target,
                pos_weight=self.question_word_weight), -1) * self.question_mask
            self.question_sequence_loss = tf.reduce_sum(self.question_sequence_loss, name='question_sequence_loss', axis=-1)
            self.question_sequence_loss = tf.reduce_mean(self.question_sequence_loss)
            
            self.loss = self.answer_loss + self.question_vector_loss + self.question_sequence_loss
            tf.summary.scalar('answer_loss', self.answer_loss)
            tf.summary.scalar('question_vector_loss', self.question_vector_loss)
            tf.summary.scalar('question_sequence_loss', self.question_sequence_loss)
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
