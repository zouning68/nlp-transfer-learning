# Copyright 2019 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""T5 Model Abstract Base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class T5Model(object):
  """Abstract Base class for T5 Model API."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def train(self, mixture_or_task_name, steps):
    raise NotImplementedError()

  @abc.abstractmethod
  def eval(self, mixture_or_task_name, checkpoint_steps, summary_dir, split):
    raise NotImplementedError()

  @abc.abstractmethod
  def predict(self, input_file, output_file, checkpoint_steps, beam_size,
              temperature):
    raise NotImplementedError()

  @abc.abstractmethod
  def finetune(self, mixture_or_task_name, finetune_steps, pretrained_model_dir,
               pretrained_checkpoint_step):
    raise NotImplementedError()
