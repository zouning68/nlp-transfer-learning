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

"""Import data modules."""
from __future__ import absolute_import

import data.mixtures
import data.postprocessors
import data.preprocessors
from data.sentencepiece_vocabulary import SentencePieceVocabulary
import data.tasks
import data.test_utils
from data.utils import *  # pylint:disable=wildcard-import
