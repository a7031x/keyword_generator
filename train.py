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
    assert np.all(np.isfinite(value)), '{} contains invalid number'.format(name)
    print('{}: {:>.4f} ~ {:>.4f}'.format(name, small, big))


def run_epoch(itr, sess, model, feeder, evaluator, writer):
    feeder.prepare('train')
    nbatch = 0
    while not feeder.eof():
        aids, qids, qv, st, kb = feeder.next()
        feed = model.feed(aids, qids, qv, st, kb)
        summary, _, seq_loss, vector_loss, loss, global_step, squeezed_logit = sess.run(
            [
                model.summary, model.optimizer, model.seq_loss, model.vector_loss, model.loss, model.global_step,
                model.squeezed_logit
            ], feed_dict=feed)
        qw = [id for id,v in enumerate(squeezed_logit[0]) if v > 0]
        writer.add_summary(summary, global_step=global_step)
        print('-----ITERATION {}, {}/{}, loss: {:>.4F} + {:>.4F}={:>.4F}'.format(itr, feeder.cursor, feeder.size, seq_loss, vector_loss, loss))
        print('keywords', feeder.qids_to_sent(qw))
        print('question', feeder.qids_to_sent([x for x in qids[0] if x != config.NULL_ID]))
        nbatch += 1
        if nbatch % 10 == 0:
            loss = evaluator.evaluate(sess, model)
            print('===================DEV loss: {:>.4F}==============='.format(loss))
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
