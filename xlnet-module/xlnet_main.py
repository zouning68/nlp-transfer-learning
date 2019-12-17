import json, os, re, sys
import tensorflow as tf
from data_utils import create_data
from xlnet_config import conf, TASK
from absl import app
from train import main as pre_train_main
from tqdm import tqdm
from run_classifier import main as classfier_main

def get_corpus():
    matchObj = re.compile(r'(.+)&([0-9]+)', re.M | re.I)
    for i, filename in enumerate(tqdm(os.listdir(conf.original_corpus))):
        if not os.path.exists(conf.pretrain_corpus): os.makedirs(conf.pretrain_corpus)
        with open(conf.pretrain_corpus + filename + ".txt", "w", encoding="utf8") as fout:
            for line in open(conf.original_corpus + filename, encoding="utf8").readlines():
                matchRes = matchObj.match(line)
                if not matchRes: continue
                text, freq = matchRes.group(1), int(matchRes.group(2))
                text = re.sub(u"[=—】★一\-【◆④\t ]{1,}|\d[、.）)．]|[(（]\d[）)]|[0-9]{3,}", "", text)
                if len(text) < 10: continue
                fout.write(text.strip().lower() + "\n")

def feature2tokens():
    with open("data/features", "r", encoding="utf8") as fin:
        features = json.load(fin)
    for feature in features:
        a=1

def get_pretrain_data():
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(create_data)

def pre_train_model():
    app.run(pre_train_main)

def run_classifier():
    app.run(classfier_main)

if __name__ == "__main__":
    #get_corpus()
    #feature2tokens()
    if TASK == 0:   get_pretrain_data()
    elif TASK == 1: pre_train_model()
    elif TASK == 2: run_classifier()