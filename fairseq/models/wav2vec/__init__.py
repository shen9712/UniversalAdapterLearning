# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import imp
from .wav2vec import *  # noqa
from .wav2vec2 import *  # noqa
from .wav2vec2_asr import *  # noqa
from .keep_mask.wav2vec2_asr_keep_mask import *  # noqa
from .keep_mask.wav2vec2_keep_mask import *  # noqa
from .random_allow.wav2vec2_asr_random_allow import *  # noqa
from .random_allow.wav2vec2_random_allow import *  # noqa
from .adapter_lang_inv.wav2vec2 import *  # noqa
from .adapter_lang_inv.wav2vec2_asr import *  # noqa
from .top_layer.wav2vec2 import *  # noqa
from .top_layer.wav2vec2_asr import *  # noqa
from .adapter.wav2vec2 import *  # noqa
from .adapter.wav2vec2_asr import *  # noqa
from .adapter_fusion.wav2vec2 import *  # noqa
from .adapter_fusion.wav2vec2_asr import *  # noqa
from .adapter_fusion_phm.wav2vec2 import *  # noqa
from .adapter_fusion_phm.wav2vec2_asr import *  # noqa
from .prefix_tune.wav2vec2 import *  # noqa
from .prefix_tune.wav2vec2_asr import *  # noqa
from .adapter_stack.wav2vec2 import *  # noqa
from .adapter_stack.wav2vec2_asr import *  # noqa
from .adapter_conv.wav2vec2 import *  # noqa
from .adapter_conv.wav2vec2_asr import *  # noqa
from .prefix_tune_langemb.wav2vec2 import *  # noqa
from .prefix_tune_langemb.wav2vec2_asr import *  # noqa
from .prefix_tune_langlogits.wav2vec2 import *  # noqa
from .prefix_tune_langlogits.wav2vec2_asr import *  # noqa
from .prefix_tune_langemb_input.wav2vec2 import *  # noqa
from .prefix_tune_langemb_input.wav2vec2_asr import *
from .prefix_tune_langemb_moa.wav2vec2 import *
from .prefix_tune_langemb_moa.wav2vec2_asr import *
from .prefix_tune_langemb_adapter.wav2vec2 import *
from .prefix_tune_langemb_adapter.wav2vec2_asr import *
from .prefix_tune_langemb_langspec_adapter.wav2vec2_asr import *
from .prefix_tune_langemb_langspec_adapter.wav2vec2 import *
from .prefix_tune_langemb_langspec.wav2vec2 import *
from .prefix_tune_langemb_langspec.wav2vec2_asr import *
from .sparse.wav2vec2 import *
from .sparse.wav2vec2_asr import *
from .adapter_parallel_gate.wav2vec2 import *
from .adapter_parallel_gate.wav2vec2_asr import *
from .adapter_sparse_fc.wav2vec2 import *
from .adapter_sparse_fc.wav2vec2_asr import *
from .prefix_tune_langemb_adapter_madg.wav2vec2 import *
from .prefix_tune_langemb_adapter_madg.wav2vec2_asr import *
from .prefix_tune_langemb_adapter_madg_share.wav2vec2 import *
from .prefix_tune_langemb_adapter_madg_share.wav2vec2_asr import *
from .adapter_side_gate.wav2vec2 import *
from .adapter_side_gate.wav2vec2_asr import *
from .prefix_tune_langemb_adapter_robust.wav2vec2 import *
from .prefix_tune_langemb_adapter_robust.wav2vec2_asr import *
from .lid_adapter_robust.wav2vec2 import *
from .lid_adapter_robust.wav2vec2_asr import *