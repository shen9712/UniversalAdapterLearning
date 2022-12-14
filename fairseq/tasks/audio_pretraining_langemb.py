# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING

from fairseq.data import AddTargetDataset, Dictionary, FileAudioDataset, encoders, FileAudioDatasetSZJ
from fairseq.data.data_utils import post_process
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig

from . import FairseqTask, register_task
from .. import utils
from ..logging import metrics


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@dataclass
class AudioPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to crop to for batching"}
    )

    # Options for reporting WER metrics during validation. Only applicable to
    # Seq2Seq models during fine-tuning
    eval_wer: bool = field(
        default=False, metadata={"help": "compute WER for Seq2Seq models"}
    )
    eval_wer_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "beam search config for evaluating wer during training"},
    )
    eval_wer_tokenizer: Any = field(
        default=None,
        metadata={"help": "tokenizer config for evaluating wer during training"},
    )
    eval_wer_post_process: str = field(
        default="letter",
        metadata={
            "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
        },
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "required for autoregressive decoders (like seq2seq models); "
            "adds 'prev_output_tokens' to input and appends eos to target"
        },
    )


@register_task("audio_pretraining_lang_emb", dataclass=AudioPretrainingConfig)
class AudioPretrainingLangEmbTask(FairseqTask):
    """"""

    cfg: AudioPretrainingConfig

    def __init__(
        self,
        cfg: AudioPretrainingConfig,
        source_dictionary=None,
        target_dictionary=None,
    ):
        super().__init__(cfg)
        self._target_dictionary = target_dictionary
        self._source_dictionary = source_dictionary
        if cfg.eval_wer:
            assert cfg.labels is not None, "eval_wer can only be set during fine-tuning"
        self.blank_symbol = "<s>"

    @classmethod
    def setup_task(cls, cfg: AudioPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        if cfg.labels:
            dict_path = os.path.join(cfg.data, f"dict.{cfg.labels}.txt")
            target_dictionary = Dictionary.load(dict_path)
        else:
            target_dictionary = None

        return cls(cfg, target_dictionary=target_dictionary)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == 'ctc'

        manifest = os.path.join(data_path, "{}.tsv".format(split))
        # self.datasets[split] = FileAudioDataset(
        #     manifest,
        #     sample_rate=task_cfg.sample_rate,
        #     max_sample_size=self.cfg.max_sample_size,
        #     min_sample_size=self.cfg.max_sample_size,
        #     min_length=self.cfg.min_sample_size,
        #     pad=task_cfg.labels is not None or task_cfg.enable_padding,
        #     normalize=task_cfg.normalize,
        # )
        self.datasets[split] = FileAudioDatasetSZJ(
            manifest,
            sample_rate=task_cfg.sample_rate,
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.max_sample_size,
            min_length=self.cfg.min_sample_size,
            pad=task_cfg.labels is not None or task_cfg.enable_padding,
            normalize=task_cfg.normalize,
            split=split,
        )

        if task_cfg.labels:
            label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
            labels = []
            with open(label_path, "r") as f:
                labels = [
                    line for i, line in enumerate(f)
                    if i in self.datasets[split].line_inds
                ]

            assert len(labels) == len(self.datasets[split]), (
                    f"labels length ({len(labels)}) and dataset length "
                    f"({len(self.datasets[split])}) do not match")

            process_label = LabelEncoder(self.target_dictionary)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=task_cfg.autoregressive,
            )

    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self._target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices

    def valid_step(self, sample, model, criterion):
        # szj ??????language id
        language_id = self.get_language_id(sample['target'])
        sample['net_input']['language_id'] = language_id
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_wer and self.cfg.autoregressive:
            metrics = self._inference_with_wer(self.sequence_generator, sample, model)
            logging_output["_num_char_errors"] = metrics["num_char_errors"]
            logging_output["_num_chars"] = metrics["num_chars"]
            logging_output["_num_word_errors"] = metrics["num_word_errors"]
            logging_output["_num_words"] = metrics["num_words"]
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)

        if self.cfg.eval_wer and self.cfg.autoregressive:
            self.sequence_generator = self.build_generator(
                [model],
                self.cfg.eval_wer_config,
            )
            if self.cfg.eval_wer_tokenizer:
                self.tokenizer = encoders.build_tokenizer(self.cfg.eval_wer_tokenizer)
            else:
                self.tokenizer = None
        return model
    
    # szj added
    # def get_language_id(self, target):
    #     """
    #     ??????target ???b x t???
    #     ??????language id ???b???
    #     0, 1, 2, 3???extra tokens
    #     4?????????
    #     5???97???????????????98???126???????????????127???157??????????????????158???184????????????
         
    #     """
    #     bsz = target.shape[0]
    #     language_id = torch.zeros(bsz, dtype=torch.int, device=target.device)

    #     mask0 = (target[:, 0] >=5).logical_and(target[:, 0] <= 97)
    #     mask1 = (target[:, 0] >=98).logical_and(target[:, 0] <= 126)
    #     mask2 = (target[:, 0] >=127).logical_and(target[:, 0] <= 157)
    #     mask3 = (target[:, 0] >=158).logical_and(target[:, 0] <= 184)
    #     mask4 = (target[:, 0] < 5)

    #     language_id = language_id.masked_fill(mask0, 0)
    #     language_id = language_id.masked_fill(mask1, 1)
    #     language_id = language_id.masked_fill(mask2, 2)
    #     language_id = language_id.masked_fill(mask3, 3)
    #     language_id = language_id.masked_fill(mask4, 4)

    #     return language_id

    def get_language_id(self, target):
        """
        ??????target ???b x t???
        ??????language id ???b???
        0, 1, 2, 3???extra tokens
        4?????????
        5???52???????????????, 53~145????????????, 146~174????????????, 175~226????????????, 227~257???????????????, 258~284????????????
        """
        bsz = target.shape[0]
        language_id = torch.zeros(bsz, dtype=torch.int, device=target.device)

        mask0 = (target[:, 0] >=5).logical_and(target[:, 0] <= 52)
        mask1 = (target[:, 0] >=53).logical_and(target[:, 0] <= 145)
        mask2 = (target[:, 0] >=146).logical_and(target[:, 0] <= 174)
        mask3 = (target[:, 0] >=175).logical_and(target[:, 0] <= 226)
        mask4 = (target[:, 0] >=227).logical_and(target[:, 0] <= 257)
        mask5 = (target[:, 0] >=258).logical_and(target[:, 0] <= 284)
        mask6 = (target[:, 0] < 5)

        language_id = language_id.masked_fill(mask0, 0)
        language_id = language_id.masked_fill(mask1, 1)
        language_id = language_id.masked_fill(mask2, 2)
        language_id = language_id.masked_fill(mask3, 3)
        language_id = language_id.masked_fill(mask4, 4)
        language_id = language_id.masked_fill(mask5, 5)
        language_id = language_id.masked_fill(mask6, 6)

        return language_id

    # def get_language_id(self, target):
    #     """
    #     ??????target ???b x t???
    #     ??????language id ???b???
    #     0, 1, 2, 3???extra tokens
    #     4?????????
    #     5???35??????????????????36???62????????????
         
    #     """
    #     bsz = target.shape[0]
    #     language_id = torch.zeros(bsz, dtype=torch.int).to(target.device)

    #     mask0 = (target[:, 0] >=5).logical_and(target[:, 0] <= 35)
    #     mask1 = (target[:, 0] >=36).logical_and(target[:, 0] <= 62)
    #     mask2 = (target[:, 0] < 5)  # ???????????????pad token????????? (???????????????); 1 1 1 1 1

    #     language_id = language_id.masked_fill(mask0, 0)
    #     language_id = language_id.masked_fill(mask1, 1)
    #     language_id = language_id.masked_fill(mask2, 2)

    #     return language_id

    # szj added
    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            # szj ??????language id
            language_id = self.get_language_id(sample['target'])
            sample['net_input']['language_id'] = language_id
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )
    
    # szj
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        # szj ??????language id
        language_id = self.get_language_id(sample['target'])
        sample['net_input']['language_id'] = language_id
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def _inference_with_wer(self, generator, sample, model):
        import editdistance

        def decode(toks):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.cfg.eval_wer_post_process,
                escape_unk=True,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        num_word_errors, num_char_errors = 0, 0
        num_chars, num_words = 0, 0
        gen_out = self.inference_step(generator, [model], sample, None)
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
            )
            num_char_errors += editdistance.eval(hyp, ref)
            num_chars += len(ref)
            hyp_words = hyp.split()
            ref_words = ref.split()
            num_word_errors += editdistance.eval(hyp_words, ref_words)
            num_words += len(ref_words)

        return {
            "num_char_errors": num_char_errors,
            "num_chars": num_chars,
            "num_word_errors": num_word_errors,
            "num_words": num_words,
        }

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.0)
        num_char_errors = sum(
            log.get("_num_char_errors", zero) for log in logging_outputs
        )
        num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
        num_word_errors = sum(
            log.get("_num_word_errors", zero) for log in logging_outputs
        )
        num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
        metrics.log_scalar("_num_char_errors", num_char_errors)
        metrics.log_scalar("_num_chars", num_chars)
        metrics.log_scalar("_num_word_errors", num_word_errors)
        metrics.log_scalar("_num_words", num_words)
        if num_words > 0:
            metrics.log_derived(
                "uer",
                lambda meters: meters["_num_char_errors"].sum
                * 100.0
                / meters["_num_chars"].sum
                if meters["_num_chars"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "wer",
                lambda meters: meters["_num_word_errors"].sum
                * 100.0
                / meters["_num_words"].sum
                if meters["_num_words"].sum > 0
                else float("nan"),
            )
