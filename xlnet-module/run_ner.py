from __future__ import absolute_import, division, print_function
from os.path import join
from absl import flags
import os, collections, json, codecs, pickle, re, xlnet
import numpy as np
import tensorflow as tf
import sentencepiece as spm
from xlnet_config import FLAGS
from data_utils import SEP_ID, VOCAB_SIZE, CLS_ID
import model_utils
import function_builder
from classifier_utils import PaddingInputExample
#from classifier_utils import convert_single_example
from prepro_utils import preprocess_text, encode_ids
from lstm_crf_layer import BLSTM_CRF
from tensorflow.contrib.layers.python.layers import initializers

import logging as logger
logger.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logger.INFO)
SEG_ID_A   = 0
SEG_ID_B   = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4
'''
# Model
flags.DEFINE_string("model_config_path", default="pretrain_model/config.json", help="Model config path.")
flags.DEFINE_float("dropout", default=0.1, help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1, help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1, help="Clamp length")
flags.DEFINE_string("summary_type", default="last", help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_bool("use_summ_proj", default=True, help="Whether to use projection for summarizing sequences.")
flags.DEFINE_bool("use_bfloat16", False, help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal", enum_values=["normal", "uniform"], help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02, help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1, help="Initialization std when init is uniform.")

# I/O paths
flags.DEFINE_bool("overwrite_data", default=False, help="If False, will use cached data if available.")
flags.DEFINE_string("init_checkpoint", default="pretrain_model/model.ckpt-35",
      help="checkpoint path for initializing the model. "
      "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("output_dir", default="proc_data/imdb", help="Output dir for TF records.")
flags.DEFINE_string("spiece_model_file", default="token_model/chinese/spiece.model", help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="finetuning_model/imdb", help="Directory for saving the finetuned model.")
flags.DEFINE_string("data_dir", default="data/aclImdb", help="Directory for input data.")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context "
      "of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("iterations", default=1000, help="number of iterations per TPU training loop.")

# training
flags.DEFINE_bool("do_train", default=True, help="whether to do training")
flags.DEFINE_integer("train_steps", default=1000, help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=500, help="number of warmup steps")
flags.DEFINE_float("learning_rate", default=2e-5, help="initial learning rate")
flags.DEFINE_float("lr_layer_decay_rate", 1.0,
                   "Top layer: lr[L] = FLAGS.learning_rate."
                   "Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("min_lr_ratio", default=0.0, help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_integer("max_save", default=0, help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=10, help="Save the model for every save_steps. If None, not to save any model.")
flags.DEFINE_integer("train_batch_size", default=32, help="Batch size for training")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-8, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")

# evaluation
flags.DEFINE_bool("do_eval", default=True, help="whether to do eval")
flags.DEFINE_bool("do_predict", default=False, help="whether to do prediction")
flags.DEFINE_float("predict_threshold", default=0, help="Threshold for binary prediction.")
flags.DEFINE_string("eval_split", default="dev", help="could be dev or test")
flags.DEFINE_integer("eval_batch_size", default=8, help="batch size for evaluation")
flags.DEFINE_integer("predict_batch_size", default=128, help="batch size for prediction.")
flags.DEFINE_string("predict_dir", default=None, help="Dir for saving prediction files.")
flags.DEFINE_bool("eval_all_ckpt", default=True, help="Eval all ckpts. If False, only evaluate the last one.")
flags.DEFINE_string("predict_ckpt", default=None, help="Ckpt path for do_predict. If None, use the last one.")

# task specific
flags.DEFINE_string("task_name", default="imdb", help="Task name")
flags.DEFINE_integer("max_seq_length", default=128, help="Max sequence length")
flags.DEFINE_integer("shuffle_buffer", default=2048, help="Buffer size used for shuffle.")
flags.DEFINE_integer("num_passes", default=1,
      help="Num passes for processing training data. "
      "This is use to batch data without loss for TPUs.")
flags.DEFINE_bool("uncased", default=False, help="Use uncased.")
flags.DEFINE_string("cls_scope", default=None, help="Classifier layer scope.")
flags.DEFINE_bool("is_regression", default=False, help="Whether it's a regression task.")

FLAGS = flags.FLAGS
'''

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_ids,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.is_real_example = is_real_example

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_data(cls, input_file):
      """Reads a BIO data."""
      with codecs.open(input_file, 'r', encoding='utf-8') as f:
          lines = []
          words = []
          labels = []
          for line in f:
              contends = line.strip()
              tokens = contends.split(' ')
              if len(tokens) == 2:
                  words.append(tokens[0])
                  labels.append(tokens[1])
              else:
                  if len(contends) == 0:
                      l = ' '.join([label for label in labels if len(label) > 0])
                      w = ' '.join([word for word in words if len(word) > 0])
                      lines.append([l, w])
                      words = []
                      labels = []
                      continue
              if contends.startswith("-DOCSTART-"):
                  words.append('')
                  continue
          return lines

#**********************************************************************************************************************#
class NerProcessor(DataProcessor):
    def __init__(self, output_dir):
        self.labels = set()
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, labels=None):
        if labels is not None:
            try:
                # 支持从文件中读取标签类型
                if os.path.exists(labels) and os.path.isfile(labels):
                    with codecs.open(labels, 'r', encoding='utf-8') as fd:
                        for line in fd:
                            self.labels.append(line.strip())
                else:
                    # 否则通过传入的参数，按照逗号分割
                    self.labels = labels.split(',')
                self.labels = set(self.labels) # to set
            except Exception as e:
                print(e)
        # 通过读取train文件获取标签的方法会出现一定的风险。
        if os.path.exists(os.path.join(self.output_dir, 'label_list.pkl')):
            with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'rb') as rf:
                self.labels = pickle.load(rf)
        else:
            if len(self.labels) > 0:
                self.labels = self.labels.union(set(["X", "[CLS]", "[SEP]"]))
                with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'wb') as rf:
                    pickle.dump(self.labels, rf)
            else:
                # self.labels = ["O", 'B-TIM', 'I-TIM', "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
                self.labels = ["O", 'B-SP', 'I-SP', "B-SS", "I-SS", "[CLS]", "[SEP]", "[PAD]"]
        return self.labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line[1]
            label = line[0]
            # if i == 0:
            #     print('label: ', label)
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        return examples

    def _read_data(self, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                if l == "0": l = "O"        # 噪音处理
                                label.append(l)
                                #self.labels.add(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    continue
            return lines

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def convert_single_example(ex_index, example, label_list, max_seq_length,
                              tokenize_fn):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[1] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_ids=[0] * max_seq_length,
        is_real_example=False)

  if label_list is not None:
    label_map = {}
    for (i, label) in enumerate(label_list):
      label_map[label] = i

  tokens_a = tokenize_fn(example.text_a)
  labels_a = example.label.split()
  tokens_b = None
  if example.text_b:
    tokens_b = tokenize_fn(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for two [SEP] & one [CLS] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for one [SEP] & one [CLS] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[:max_seq_length - 2]

  tokens = []
  segment_ids = []
  label_ids = []
  for i, token in enumerate(tokens_a):
    tokens.append(token)
    segment_ids.append(SEG_ID_A)
    label_ids.append(label_map[labels_a[i]])
  tokens.append(SEP_ID)
  segment_ids.append(SEG_ID_A)
  label_ids.append(label_map["[SEP]"])

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(SEG_ID_B)
    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_B)

  tokens.append(CLS_ID)
  segment_ids.append(SEG_ID_CLS)
  label_ids.append(label_map["[CLS]"])

  input_ids = tokens

  # The mask has 0 for real tokens and 1 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [0] * len(input_ids)

  # Zero-pad up to the sequence length.
  if len(input_ids) < max_seq_length:
    delta_len = max_seq_length - len(input_ids)
    input_ids = [0] * delta_len + input_ids
    input_mask = [1] * delta_len + input_mask
    segment_ids = [SEG_ID_PAD] * delta_len + segment_ids
    label_ids = [label_map["[PAD]"]] * delta_len + label_ids

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(label_ids) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_ids=label_ids)
  return feature

#**********************************************************************************************************************#
def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenize_fn, output_file,
    num_passes=1):
  """Convert a set of `InputExample`s to a TFRecord file."""

  # do not create duplicated records
  if tf.gfile.Exists(output_file) and not FLAGS.overwrite_data:
    tf.logging.info("Do not overwrite tfrecord {} exists.".format(output_file))
    return

  tf.logging.info("Create new tfrecord {}.".format(output_file))

  writer = tf.python_io.TFRecordWriter(output_file)

  if num_passes > 1:
    examples *= num_passes

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example {} of {}".format(ex_index,
                                                        len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenize_fn)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_float_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)
    features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()

def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  tf.logging.info("Input tfrecord file {}".format(input_file))

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def input_fn(params, input_context=None):
    """The actual input function."""
    if FLAGS.use_tpu:
      batch_size = params["batch_size"]
    elif is_training:
      batch_size = FLAGS.train_batch_size
    elif FLAGS.do_eval:
      batch_size = FLAGS.eval_batch_size
    else:
      batch_size = FLAGS.predict_batch_size

    d = tf.data.TFRecordDataset(input_file)
    # Shard the dataset to difference devices
    if input_context is not None:
      tf.logging.info("Input pipeline id %d out of %d",
          input_context.input_pipeline_id, input_context.num_replicas_in_sync)
      d = d.shard(input_context.num_input_pipelines,
                  input_context.input_pipeline_id)

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
      d = d.repeat()

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def get_model_fn(n_class):
  def model_fn(features, labels, mode, params):
    #### Training or Evaluation
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #### Get loss from inputs
#********************************************************************************************#
    bsz_per_core = tf.shape(features["input_ids"])[0]
    inp = tf.transpose(features["input_ids"], [1, 0])
    seg_id = tf.transpose(features["segment_ids"], [1, 0])
    inp_mask = tf.transpose(features["input_mask"], [1, 0])
    label_ids = features["label_ids"]

    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training, True, FLAGS)
    xlnet_model = xlnet.XLNetModel(
        xlnet_config=xlnet_config,
        run_config=run_config,
        input_ids=inp,
        seg_ids=seg_id,
        input_mask=inp_mask)
    #summary = xlnet_model.get_pooled_out(FLAGS.summary_type, FLAGS.use_summ_proj)
    # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    xlnet_model_out = xlnet_model.get_sequence_output()
    embedding = tf.transpose(xlnet_model_out, [1, 0, 2])
    max_seq_length = embedding.shape[1].value
    # 算序列真实长度
    used = tf.sign(tf.abs(features["input_ids"]))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
    # 添加CRF output layer
    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=10, cell_type="lstm", num_layers=1,
                          dropout_rate=0.5, initializers=initializers, num_labels=n_class,
                          seq_length=max_seq_length, labels=label_ids, lengths=lengths, is_training=is_training)
    total_loss, logits, trans, pred_ids = blstm_crf.add_blstm_crf_layer(crf_only=True)
#********************************************************************************************#

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    #### load pretrained models
    scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

    #### Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(label_ids, pred_ids):
          return {
              "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids),
          }
      eval_metrics = metric_fn(features["label_ids"], pred_ids)
      eval_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metrics
      )
      return eval_spec
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "logits": logits,
          "labels": label_ids,
          "pred_ids": pred_ids,
          "input_mask": features["input_mask"]
          }
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions
      )
      return output_spec

    #### Configuring the optimizer
    train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)

    monitor_dict = {}
    monitor_dict["lr"] = learning_rate

    #### Constucting training TPUEstimatorSpec with new cache.
    train_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
    return train_spec
  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  #### Validate flags
  if FLAGS.save_steps is not None:
    FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

  if FLAGS.do_predict:
    predict_dir = FLAGS.predict_dir
    if not tf.gfile.Exists(predict_dir):
      tf.gfile.MakeDirs(predict_dir)

  processors = {
      "ner": NerProcessor
  }

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval, `do_predict` or "
        "`do_submit` must be True.")

  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name](FLAGS.output_dir)
  label_list = processor.get_labels()

  sp = spm.SentencePieceProcessor()
  sp.Load(FLAGS.spiece_model_file)
  def tokenize_fn(text):
    text = preprocess_text(text, lower=FLAGS.uncased, remove_space=True, keep_accents=False)
    pieces = text.split()
    ids = [sp.PieceToId(piece) for piece in pieces]
    return ids

  run_config = model_utils.configure_tpu(FLAGS)

  model_fn = get_model_fn(len(label_list))

  spm_basename = os.path.basename(FLAGS.spiece_model_file)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  if FLAGS.use_tpu:
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)
  else:
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

  if FLAGS.do_train:
    train_file_base = "{}.len-{}.train.tf_record".format(
        spm_basename, FLAGS.max_seq_length)
    train_file = os.path.join(FLAGS.output_dir, train_file_base)
    tf.logging.info("Use tfrecord file {}".format(train_file))

    train_examples = processor.get_train_examples(FLAGS.data_dir)
    np.random.shuffle(train_examples)
    tf.logging.info("Num of train samples: {}".format(len(train_examples)))

    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenize_fn,
        train_file, FLAGS.num_passes)

    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)

    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

  if FLAGS.do_eval or FLAGS.do_predict:
    if FLAGS.eval_split == "dev":
      eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    else:
      eval_examples = processor.get_test_examples(FLAGS.data_dir)

    tf.logging.info("Num of eval samples: {}".format(len(eval_examples)))

  if FLAGS.do_eval:
    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on. These do NOT count towards the metric (all tf.metrics
    # support a per-instance weight, and these get a weight of 0.0).
    #
    # Modified in XL: We also adopt the same mechanism for GPUs.
    while len(eval_examples) % FLAGS.eval_batch_size != 0:
      eval_examples.append(PaddingInputExample())

    eval_file_base = "{}.len-{}.{}.eval.tf_record".format(
        spm_basename, FLAGS.max_seq_length, FLAGS.eval_split)
    eval_file = os.path.join(FLAGS.output_dir, eval_file_base)

    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenize_fn,
        eval_file)

    assert len(eval_examples) % FLAGS.eval_batch_size == 0
    eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=True)

    # Filter out all checkpoints in the directory
    steps_and_files = []
    filenames = tf.gfile.ListDirectory(FLAGS.model_dir)

    for filename in filenames:
      if filename.endswith(".index"):
        ckpt_name = filename[:-6]
        cur_filename = join(FLAGS.model_dir, ckpt_name)
        global_step = int(cur_filename.split("-")[-1])
        tf.logging.info("Add {} to eval list.".format(cur_filename))
        steps_and_files.append([global_step, cur_filename])
    steps_and_files = sorted(steps_and_files, key=lambda x: x[0])

    # Decide whether to evaluate all ckpts
    if not FLAGS.eval_all_ckpt:
      steps_and_files = steps_and_files[-1:]

    eval_results = []
    for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
      ret = estimator.evaluate(
          input_fn=eval_input_fn,
          steps=eval_steps,
          checkpoint_path=filename)

      ret["step"] = global_step
      ret["path"] = filename

      eval_results.append(ret)

      tf.logging.info("=" * 80)
      log_str = "Eval result | "
      for key, val in sorted(ret.items(), key=lambda x: x[0]):
        log_str += "{} {} | ".format(key, val)
      tf.logging.info(log_str)

    eval_results.sort(key=lambda x: x["loss"], reverse=True)

    tf.logging.info("=" * 80)
    log_str = "Best result | "
    for key, val in sorted(eval_results[0].items(), key=lambda x: x[0]):
      log_str += "{} {} | ".format(key, val)
    tf.logging.info(log_str)

  if FLAGS.do_predict:
    eval_file_base = "{}.len-{}.{}.predict.tf_record".format(
        spm_basename, FLAGS.max_seq_length, FLAGS.eval_split)
    eval_file = os.path.join(FLAGS.output_dir, eval_file_base)

    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenize_fn,
        eval_file)

    pred_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    predict_results = []
    with tf.gfile.Open(os.path.join(predict_dir, "{}.tsv".format(
        task_name)), "w") as fout:
      fout.write("index\tprediction\n")

      for pred_cnt, result in enumerate(estimator.predict(
          input_fn=pred_input_fn,
          yield_single_examples=True,
          checkpoint_path=FLAGS.predict_ckpt)):
        if pred_cnt % 1000 == 0:
          tf.logging.info("Predicting submission for example: {}".format(
              pred_cnt))

        pred_ids = [int(x) for x in result["pred_ids"].flat]
        input_mask = [int(x) for x in result["input_mask"].flat]
        label_out = [label_list[pred_ids[i]] for i in range(len(input_mask)) if input_mask[i] != 1]
        predict_results.append(label_out)
        fout.write("{}\t{}\n".format(pred_cnt, label_out))

    predict_json_path = os.path.join(predict_dir, "{}.logits.json".format(
        task_name))

    with tf.gfile.Open(predict_json_path, "w") as fp:
      json.dump(predict_results, fp, indent=4)


if __name__ == "__main__":
  tf.app.run()
