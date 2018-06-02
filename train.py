import tensorflow as tf
import config
import utils
import numpy as np
from data import TrainFeeder, Dataset
from model import Model


def run_epoch(type, sess, model, feeder, writer):
    feeder.prepare(type)
    while not feeder.eof():
        aids, qv, av, kb = feeder.next(32)
        feed = model.feed(aids, qv, av, kb)
        summary, _, loss, global_step = sess.run([model.summary, model.optimizer, model.loss, model.global_step], feed_dict=feed)
        model.save(sess)
        writer.add_summary(summary, global_step=global_step)
        print('loss: {:>.4F}'.format(loss))


def train(auto_stop):
    model = Model(config.checkpoint_folder)
    dataset = Dataset()
    feeder = TrainFeeder(dataset)
    with tf.Session() as sess:
        model.restore(sess)
        #utils.rmdir(config.log_folder)
        writer = tf.summary.FileWriter(config.log_folder, sess.graph)
        model.summarize(writer)
        while True:
            run_epoch('train', sess, model, feeder, writer)


train(False)
