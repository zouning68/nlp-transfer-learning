import json
import tensorflow as tf
from data_utils import create_data
from xlnet_config import conf

def jd2corpus():
    jd = [line.split("\t")[33].replace("\\n","\n") for line in open(conf.jd, encoding="utf8").readlines()]
    with open(conf.jd_data, "w", encoding="utf8") as fin:
        for e in jd: fin.write(e+"\n")

def feature2tokens():
    with open("data/features", "r", encoding="utf8") as fin:
        features = json.load(fin)
    for feature in features:
        a=1

def get_pretrain_data():
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(create_data)

if __name__ == "__main__":
    pass
    #jd2corpus()
    #feature2tokens()
    #get_pretrain_data()