import numpy as np
import preprocess
import utils
import config
from data import TrainFeeder, align2d, Dataset


def run_epoch(sess, model, feeder, writer):
    feeder.prepare('dev')
    while not feeder.eof():
        aids, qv, av, kb = feeder.next(32)
        feed = model.feed(aids, qv, av, kb)
        answer_logit, question_logit = sess.run([model.answer_logit, model.question_logit], feed_dict=feed)
        question = [id for id, v in enumerate(question_logit) if v >= 0]
        answer = [id for id, v in enumerate(answer_logit) if v >= 0]
        return question, answer
        

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class Evaluator(TrainFeeder):
    def __init__(self, dataset=None):
        super(Evaluator, self).__init__(Dataset() if dataset is None else dataset)


    def create_feed(self, answer, question):
        #question = question.split(' ')
        #answer = answer.split(' ')
        aids = self.asent_to_id(answer)
        qids = self.qsent_to_id(question)
        qv, _ = self.label_qa(question)
        st = self.seq_tag(question, answer)
        return aids, qids, qv, st, 1.0


    def predict(self, sess, model, answer, question):
        aids, qids, qv, av, kb = self.create_feed(answer, question)
        feed = model.feed([aids], [qids], [qv], [av], kb)
        answer_logit, question_logit, qsl = sess.run([model.answer_logit, model.question_logit, model.question_sequence_logit], feed_dict=feed)
        #question_ids = [id for id, v in enumerate(question_logit[0]) if v >= 0]
        #answer_ids = [id for id, v in enumerate(answer_logit[0]) if v >= 0]
        qids = sorted(enumerate(question_logit[0]), key=lambda x:-x[1])[:10]
        aw = set([word for word,value in zip(answer, answer_logit[0]) if value >= 0])
        qw = set(self.qids_to_sent([id for id,_ in qids]))
        qs = self.qids_to_sent(np.argmax(qsl[0], axis=-1))
        print('==================================================')
        print('answer', ' '.join(answer))
        print('---------------------------------------------------')
        print('question', ' '.join(question))
        print('words', qw, aw)
        print('predict question', qs)
        #print('question score', [v for _,v in qids])
        #print('answer score', ['{}:{:>.4f}'.format(w,x) for w,x in zip(answer, answer_logit[0])])
        return qw, aw 


    def evaluate(self, sess, model):
        self.prepare('dev')
        aids, qids, qv, st, kb = self.next(64)
        feed = model.feed(aids, qids, qv, st, kb)
        loss = sess.run(model.loss, feed_dict=feed)
        return loss


if __name__ == '__main__':
    from model import Model
    import tensorflow as tf
    evaluator = Evaluator()
    model = Model(evaluator.dataset.qi2c, config.checkpoint_folder)
    with tf.Session() as sess:
        model.restore(sess)
        #evaluator.evaluate(sess, model, 'The cat sat on the mat', 'what is on the mat')
        evaluator.prepare('dev')
        for question, answer in evaluator.data[:10]:
            evaluator.predict(sess, model, answer, question)