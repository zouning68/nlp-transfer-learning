"""Main file for launching training/eval/predictions of mesh-transformer model."""
from __future__ import absolute_import, division, print_function
import importlib, os, sys, gin, pkg_resources
from absl import app
from mesh_tensorflow.transformer import utils
from models import mtf_model
import tensorflow.compat.v1 as tf
from t5_config import FLAGS
import data

def main(_):
  if FLAGS.module_import:
    for module in FLAGS.module_import:
      importlib.import_module(module)

  if FLAGS.t5_tfds_data_dir:
    data.set_tfds_data_dir_override(FLAGS.t5_tfds_data_dir)
  data.add_global_cache_dirs(FLAGS.additional_task_cache_dirs)

  # Add search path for gin files stored in package.
  gin.add_config_file_search_path(pkg_resources.resource_filename(__name__, "gin"))

  tf.io.gfile.makedirs(FLAGS.model_dir)
  suffix = 0
  command_filename = os.path.join(FLAGS.model_dir, "command")
  while tf.io.gfile.exists(command_filename):
    suffix += 1
    command_filename = os.path.join(
        FLAGS.model_dir, "command.{}".format(suffix))
  with tf.io.gfile.GFile(command_filename, "w") as f:
    f.write(" ".join(sys.argv))

  utils.parse_gin_defaults_and_flags()

  if FLAGS.use_model_api:
    model = mtf_model.MtfModel(
        tpu_job_name=FLAGS.tpu_job_name,
        tpu=FLAGS.tpu,
        gcp_project=FLAGS.gcp_project,
        tpu_zone=FLAGS.tpu_zone,
        model_dir=FLAGS.model_dir,
        batch_size=1024
    )

    if FLAGS.checkpoint_mode == "latest":
      checkpoint_steps = -1
    elif FLAGS.checkpoint_mode == "all":
      checkpoint_steps = "all"
    else:
      checkpoint_steps = [int(c) for c in FLAGS.checkpoint_steps]

    if FLAGS.mode == "train":
      model.train(mixture_or_task_name=FLAGS.mixture_or_task,
                  steps=FLAGS.train_steps)
    elif FLAGS.mode == "eval":
      model.eval(mixture_or_task_name=FLAGS.mixture_or_task,
                 checkpoint_steps=checkpoint_steps,
                 summary_dir=FLAGS.eval_summary_dir,
                 split=FLAGS.eval_split)
    elif FLAGS.mode == "finetune":
      assert (FLAGS.checkpoint_mode == "latest" or
              (FLAGS.checkpoint_mode == "specific" and
               len(FLAGS.checkpoint_steps) == 1)), \
          "Must specify a single checkpoint for finetuning a model."

      if isinstance(checkpoint_steps, list):
        checkpoint_steps = checkpoint_steps[0]

      model.finetune(
          mixture_or_task_name=FLAGS.mixture_or_task,
          steps=FLAGS.train_steps,
          pretrained_model_dir=FLAGS.pretrained_model_dir,
          checkpoint_steps=checkpoint_steps)
    else:
      model.predict(
          checkpoint_steps=checkpoint_steps,
          input_file=FLAGS.input_file,
          output_file=FLAGS.output_file)

  else:
    utils.run(
        tpu_job_name=FLAGS.tpu_job_name,
        tpu=FLAGS.tpu,
        gcp_project=FLAGS.gcp_project,
        tpu_zone=FLAGS.tpu_zone,
        model_dir=FLAGS.model_dir)

def console_entry_point():
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)

if __name__ == "__main__":
  console_entry_point()
