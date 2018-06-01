import tensorflow as tf
import utils
import func
import config

class Model(object):
    def __init__(self, question_vocab_size, answer_vocab_size, ckpt_folder, name='model'):
        self.name = name
        self.ckpt_folder = ckpt_folder
        self.question_vocab_size = question_vocab_size
        self.answer_vocab_size = answer_vocab_size
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
        self.create_attention()


    def create_inputs(self):
        with tf.name_scope('input'):
            self.input_word = tf.placeholder(tf.int32, shape=[None, None], name='word')
            self.input_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.batch_size = tf.shape(self.input_word)[0]
            self.mask, self.length = func.tensor_to_mask(self.input_word)
            self.input_label_answer = tf.placeholder(tf.float32, shape=[None, self.answer_vocab_size], name='label_answer')
            self.input_label_question = tf.placeholder(tf.float32, shape=[None, self.question_vocab_size], name='label_question')


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
            self.encoding, _ = func.rnn('bi-lstm', self.emb, self.length, config.encoder_hidden_dim, 2, self.input_keep_prob)
            self.encoding = tf.nn.dropout(self.encoding, self.input_keep_prob, name='encoding')
            b_fw = self.encoding[:,-1,:config.encoder_hidden_dim]
            b_bw = self.encoding[:,0,config.encoder_hidden_dim:]
            self.encoder_last_state = tf.concat([b_fw, b_bw], -1)
            tf.summary.histogram('encoder/encoding', self.encoding)
            tf.summary.histogram('encoder/last_state', self.encoder_last_state)


    def create_decoder(self):
        with tf.name_scope('decoder'):
            


    def create_attention(self):
        with tf.name_scope('attention'):
            self.ct = func.dot_attention(self.decoder_h, self.passage_enc, self.passage_mask, config.dot_attention_dim, self.input_keep_prob)
            self.combined_h = tf.concat([self.decoder_h, self.ct], -1, name='combined_h')#[batch, question_len, 450]           
            self.wt = tf.get_variable('wt', shape=[config.max_question_len, self.combined_h.get_shape()[-1], config.decoder_hidden_dim])
            self.ws = tf.get_variable('ws', shape=[config.decoder_hidden_dim, self.vocab_size])
            question_len = tf.shape(self.combined_h)[1]
            self.wt_h = tf.einsum('bij,ijk->bik', self.combined_h, self.wt[:question_len,:,:], name='wt_h')
            self.ws_tanh_wt = tf.einsum('bik,kj->bij', tf.tanh(self.wt_h), self.ws)


if __name__ == '__main__':
    model = Model(10000, None)