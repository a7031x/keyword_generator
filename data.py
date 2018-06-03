import config
import utils
import random

class Dataset(object):
    def __init__(self):
        self.aw2i, self.ai2w, self.ai2c = load_vocab(config.answer_vocab_file, config.answer_vocab_size)
        self.qw2i, self.qi2w, self.qi2c = load_vocab(config.question_vocab_file, config.question_vocab_size)        
        self.train_set = load_qa(config.train_file, config.answer_limit)
        self.dev_set = load_qa(config.dev_file, config.answer_limit)


class Feeder(object):
    def __init__(self, dataset):
        self.dataset = dataset


    def qword_to_id(self, word):
        if word in self.dataset.qw2i:
            return self.dataset.qw2i[word]
        else:
            return config.OOV_ID


    def aword_to_id(self, word):
        if word in self.dataset.aw2i:
            return self.dataset.aw2i[word]
        else:
            return config.OOV_ID


    def asent_to_id(self, sent):
        return [self.aword_to_id(w) for w in sent]


    def qsent_to_id(self, sent):
        return [self.qword_to_id(w) for w in sent]


    def aids_to_sent(self, ids):
        return [self.dataset.ai2w[id] for id in ids]


    def qids_to_sent(self, ids):
        return [self.dataset.qi2w[id] for id in ids]


    def label_qa(self, sent):
        qv = [0] * len(self.dataset.qw2i)
        av = [0] * len(self.dataset.aw2i)
        for word in sent:
            if word in self.dataset.qw2i:
                qv[self.dataset.qw2i[word]] = 1
            if word in self.dataset.aw2i:
                av[self.dataset.aw2i[word]] = 1
        return qv, av


    def seq_tag(self, question, answer):
        return [1 if word in question else 0 for word in answer]


class TrainFeeder(Feeder):
    def __init__(self, dataset):
        super(TrainFeeder, self).__init__(dataset)


    def prepare(self, type):
        if type == 'train':
            self.data = self.dataset.train_set
            self.keep_prob = 0.7
            random.shuffle(self.data)
        elif type == 'dev':
            self.data = self.dataset.dev_set
            self.keep_prob = 1.0
        self.cursor = 0
        self.size = len(self.data)


    def eof(self):
        return self.cursor == self.size


    def next(self, batch_size=64):
        size = min(self.size - self.cursor, batch_size)
        batch = self.data[self.cursor:self.cursor+size]
        q, a = zip(*batch)
        qids, aids = [self.qsent_to_id(x) for x in q], [self.asent_to_id(x) for x in a]
        qa_vector = [self.label_qa(x) for x in q]
        q_vector, _ = zip(*qa_vector)
        seq_tag = [self.seq_tag(question, answer) for question,answer in zip(q,a)]
        self.cursor += size
        return align2d(aids), align2d(qids), q_vector, align2d(seq_tag), self.keep_prob


def load_vocab(filename, count):
    w2i = {
        config.NULL: config.NULL_ID,
        config.OOV: config.OOV_ID,
        config.SOS: config.SOS_ID,
        config.EOS: config.EOS_ID
    }
    i2c = {}
    all_entries = list(utils.read_all_lines(filename))
    count -= len(w2i)
    count = min(count, len(all_entries))
    for line in all_entries[:count]:
        word, freq = line.rsplit(':', 1)
        id = len(w2i)
        w2i[word] = id
        i2c[id] = freq
    i2w = {k:v for v,k in w2i.items()}
    i2c[config.OOV_ID] = len(all_entries) - count
    return w2i, i2w, i2c


def load_qa(filename, answer_limit=0):
    q = None
    qas = []
    for line in utils.read_all_lines(filename):
        if q is not None:
            answer = line.split(' ')
            if answer_limit == 0 or len(answer) <= answer_limit:
                qas.append((q.split(' ')[:config.max_question_len], answer))
            q = None
        else:
            q = line
    return qas


def align2d(values, fill=0):
    mlen = max([len(row) for row in values])
    return [row + [fill] * (mlen - len(row)) for row in values]


def align3d(values, fill=0):
    lengths = [[len(x) for x in y] for y in values]
    maxlen0 = max([max(x) for x in lengths])
    maxlen1 = max([len(x) for x in lengths])
    for row in values:
        for line in row:
            line += [fill] * (maxlen0 - len(line))
        row += [([fill] * maxlen0)] * (maxlen1 - len(row))
    return values


if __name__ == '__main__':
    dataset = Dataset()
    feeder = TrainFeeder(dataset)
    feeder.prepare('dev')
    aids, q, a, keep_prob = feeder.next()
    for question, answer in zip(q, a):
        print('--------------------')
        qw = [id for id, flag in enumerate(question) if flag == 1]
        aw = [id for id, flag in enumerate(answer) if flag == 1]
        qw = feeder.qids_to_sent(qw)
        aw = feeder.aids_to_sent(aw)
        print('question', qw)
        print('answer', aw)
