import json
import os
import utils
import config
import re
import data
from collections import defaultdict

stop_words = set(utils.read_all_lines(config.stopwords_file))

def create_question_vocab(filename):
    vocab = defaultdict(lambda: 0)
    for line in utils.read_all_lines(filename):
        sample = json.loads(line)
        question = sample['segmented_question']
        for word in question:
            vocab[word] += 1
    vocab = sorted(vocab.items(), key=lambda x:-x[1])
    utils.write_all_lines(config.question_vocab_file, ['{}:{}'.format(w,c) for w,c in vocab])

    
def create_answer_vocab(filename):
    vocab = defaultdict(lambda: 0)
    for line in utils.read_all_lines(filename):
        sample = json.loads(line)
        for doc in sample['documents']:
            for answer in doc['segmented_paragraphs']:
                for word in answer:
                    vocab[word] += 1
    vocab = sorted(vocab.items(), key=lambda x:-x[1])
    utils.write_all_lines(config.answer_vocab_file, ['{}:{}'.format(w,c) for w,c in vocab])


def create_vocab(filename):
    qv = defaultdict(lambda: 0)
    av = defaultdict(lambda: 0)
    qset = set()
    aset = set()
    for q, a in data.load_qa(filename):
        sq = str.join('', q)
        sa = str.join('', a)
        if sq not in qset:
            for word in q:
                qv[word] += 1
            qset.add(sq)
        if sa not in aset:
            for word in a:
                av[word] += 1
            aset.add(sa)
    qv = sorted(qv.items(), key=lambda x:-x[1])
    av = sorted(av.items(), key=lambda x:-x[1])
    utils.write_all_lines(config.question_vocab_file, ['{}:{}'.format(w,c) for w,c in qv])
    utils.write_all_lines(config.answer_vocab_file, ['{}:{}'.format(w,c) for w,c in av])
    utils.write_all_lines('./generate/questions.txt', qset)
    utils.write_all_lines('./generate/answers.txt', aset)


def prepare_dataset_with_document(source, target):
    aqs = []
    all = 0
    for line in utils.read_all_lines(source):
        sample = json.loads(line)
        question = sample['segmented_question']
        question_words = set(question) - stop_words
        for doc in sample['documents']:
            for answer in doc['segmented_paragraphs']:
                answer_words = set(answer) - stop_words
                common = question_words & answer_words
                if len(common) / len(question_words) > 0.3:
                    a = rip_marks(str.join(' ', answer))
                    q = rip_marks(str.join(' ', question))
                    if len(a) > 2 * len(q):
                        aqs.append((a, q))
                all += 1
    print('{}: {}/{} preprocessed'.format(source, len(aqs), all))
    #utils.save_json(target, [{'q': q, 'a': a} for a,q in aqs])
    utils.write_all_lines(target, ['{}\n{}\n'.format(q,a) for a,q in aqs])
    return aqs


def rip_marks(text):
    r = text
#    r = re.sub(r'< (ul|li|p|(/ .*)) >', r'', text)
#    r = re.sub(r'& (.*) ;', r'', r)
#    r = r.replace('\\', '')
    r = r.replace('\t', ' ')
    r = re.sub(r'[ ]+', r' ', r)
    r = r.strip()
    return r


if __name__ == '__main__':
    prepare_dataset_with_document(config.raw_train_file, config.train_file)
    prepare_dataset_with_document(config.raw_dev_file, config.dev_file)
    create_vocab(config.train_file)
