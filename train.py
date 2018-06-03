import tensorflow as tf
import config
import utils
import numpy as np
from evaluator import Evaluator
from data import TrainFeeder, Dataset
from model import Model


def diagm(name, value):
    small = np.min(value)
    big = np.max(value)
    print('{}: {:>.4f} ~ {:>.4f}'.format(name, small, big))


def run_epoch(itr, sess, model, feeder, evaluator, writer):
    feeder.prepare('train')
    nbatch = 0
    while not feeder.eof():
        aids, qids, qv, st, kb = feeder.next()
        feed = model.feed(aids, qids, qv, st, kb)
        summary, _, loss, global_step, answer_logit, self_match, reg_loss = sess.run(
            [
                model.summary, model.optimizer, model.loss, model.global_step,
                model.answer_logit, model.self_match, model.regularization_loss
            ], feed_dict=feed)
        writer.add_summary(summary, global_step=global_step)
        diagm('answer_logit', answer_logit)
        diagm('self_match', self_match)
        diagm('reg_loss', reg_loss)
        print('----ITERATION {}, {}/{}, loss: {:>.4F}'.format(itr, feeder.cursor, feeder.size, loss))
        nbatch += 1
        if nbatch % 10 == 0:
            loss = evaluator.evaluate(sess, model)
            print('====DEV loss: {:>.4F}----'.format(loss))
            model.save(sess)


def train(auto_stop):
    dataset = Dataset()
    feeder = TrainFeeder(dataset)
    evaluator = Evaluator(dataset)
    model = Model(dataset.qi2c, config.checkpoint_folder)
    with tf.Session() as sess:
        model.restore(sess)
        #utils.rmdir(config.log_folder)
        writer = tf.summary.FileWriter(config.log_folder, sess.graph)
        model.summarize(writer)
        itr = 0
        while True:
            itr += 1
            run_epoch(itr, sess, model, feeder, evaluator, writer)


train(False)
