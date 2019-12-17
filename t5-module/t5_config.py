from absl import flags
'''
flags.DEFINE_multi_string("gin_file", "dataset.gin", "Path to a Gin file.")
flags.DEFINE_multi_string("gin_file", "gs://t5-data/pretrained_models/small/operative_config.gin", "Path to a Gin file.")
flags.DEFINE_multi_string("gin_param", "utils.tpu_mesh_shape.model_parallelism = 1", "Gin parameter binding.")
flags.DEFINE_multi_string("gin_param", "utils.tpu_mesh_shape.tpu_topology = '2x2'", "Gin parameter binding.")
flags.DEFINE_multi_string("gin_param", "MIXTURE_NAME = 'glue_mrpc_v002'", "Gin parameter binding.")
flags.DEFINE_list("gin_location_prefix", [], "Gin file search path.")
'''
flags.DEFINE_string(
    "tpu_job_name", None,
    "Name of TPU worker binary. Only necessary if job name is changed from "
    "default tpu_worker.")

flags.DEFINE_string("model_dir", "transformer_standalone", "Estimator model_dir")

flags.DEFINE_string(
    "tpu", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")

flags.DEFINE_string(
    "gcp_project",
    None,
    "Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_string(
    "tpu_zone", None,
    "GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_multi_string(
    "module_import", None,
    "Modules to import. Use this, for example, to add new `Task`s to the "
    "global `TaskRegistry`.")

flags.DEFINE_string(
    "t5_tfds_data_dir", "tfds_data",
    "If set, this directory will be used to store datasets prepared by "
    "TensorFlow Datasets that are not available in the public TFDS GCS bucket. "
    "Note that this flag overrides the `tfds_data_dir` attribute of all "
    "`Task`s.")

flags.DEFINE_list("additional_task_cache_dirs", [], "Directories to search for Tasks in addition to defaults.")

flags.DEFINE_boolean("use_model_api", True, "Use model API instead of utils.run.")

# Note: All the args from here on are only used when use_model_api is set
flags.DEFINE_enum("mode", "train", ["train", "eval", "predict", "finetune"], "Mode with which to run the model.")

# Train mode args
flags.DEFINE_integer("train_steps", 1000, "Number of training iterations.")

flags.DEFINE_string("mixture_or_task", "trivia_qa_v010", "Name of Mixture or Task to use for training/evaluation.")  # wmt_t2t_ende_v003
flags.DEFINE_string("pretrained_model_dir", "", "Pretrained model dir for finetuning a model.")

# Eval mode args
flags.DEFINE_enum(
    "checkpoint_mode", "all", ["all", "latest", "specific"],
    "Checkpoint steps to use when running 'eval', 'predict', and 'finetune' "
    "modes. Can specify a list of checkpoints or all or the latest checkpoint. "
    "'finetune' mode works with 'latest' or 'specific' with a single "
    "checkpoint.")

flags.DEFINE_list(
    "checkpoint_steps", [],
    "Checkpoint step numbers used for 'eval', 'predict', and 'finetune' modes. "
    "This argument is only used when which_checkpoint='specific'. "
    "For the 'finetune' mode, only a single checkpoint must be specified.")

flags.DEFINE_string("eval_summary_dir", "", "Path to save eval summaries")
flags.DEFINE_string("eval_split", "validation", "Dataset split to use for evaluation.")

# Predict mode args
flags.DEFINE_string("input_file", "", "Path to input file for decoding.")
flags.DEFINE_string("output_file", "", "Path to output file to save decodes.")

FLAGS = flags.FLAGS