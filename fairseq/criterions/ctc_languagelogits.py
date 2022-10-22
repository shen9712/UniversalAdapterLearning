# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round


@dataclass
class CtcCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )


@register_criterion("ctc_lang_logits", dataclass=CtcCriterionConfig)
class CtcCriterion(FairseqCriterion):
    def __init__(self, cfg: CtcCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

        # szj
        self.lang_ce_loss = torch.nn.CrossEntropyLoss(reduction='sum')

    def get_language_id(self, target):
        """
        根据target （b x t）
        得到language id （b）
        0, 1, 2, 3为extra tokens
        4为空格
        5～97为越南语，98～126为海地语，127～157为库尔德语，158～184为皮金语
         
        """
        bsz = target.shape[0]
        language_id = torch.zeros(bsz, dtype=torch.int).cuda()

        mask0 = (target[:, 0] >=5).logical_and(target[:, 0] <= 97)
        mask1 = (target[:, 0] >=98).logical_and(target[:, 0] <= 126)
        mask2 = (target[:, 0] >=127).logical_and(target[:, 0] <= 157)
        mask3 = (target[:, 0] >=158).logical_and(target[:, 0] <= 184)
        mask4 = (target[:, 0] < 5)

        language_id = language_id.masked_fill(mask0, 0)
        language_id = language_id.masked_fill(mask1, 1)
        language_id = language_id.masked_fill(mask2, 2)
        language_id = language_id.masked_fill(mask3, 3)
        language_id = language_id.masked_fill(mask4, 4)

        return language_id

    def get_language_class_loss(self, logits, language_id):
        """
        logits: b x 1 x 5
        language_id: b
        """
        loss = self.lang_ce_loss(logits.squeeze(1), language_id.to(torch.long))

        return loss

    
    def get_lang_correct(self, logits, language_id):
        """
        logits: b x 1 x 5
        language_id: b
        """
        prob = utils.softmax(logits.squeeze(1).float(), dim=-1)
        pred = torch.argmax(prob, dim=-1)
        correct = (language_id == pred).sum()

        return correct


    def forward(self, model, sample, reduce=True):
        # szj 得到language id
        language_id = self.get_language_id(sample['target'])
        # sample['net_input']['language_id'] = language_id

        net_output = model(**sample["net_input"])
        # szj 只更新lid模块
        if net_output['encoder_out'] is None:
            return self.forward_lid(model, sample, net_output, language_id, reduce)

        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            non_padding_mask = ~net_output["padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss_ctc = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )
        
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens

        # szj 将句子级别的lang loss归到token级别
        # sample size, nsentences 在这套代码中都表示batch中的句子数
        loss_lang = self.get_language_class_loss(net_output['language_logits'], language_id)  * (ntokens / sample["id"].numel()) 
        loss = loss_ctc  # + loss_lang

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "loss_ctc": utils.item(loss_ctc.data),
            "loss_lang": utils.item(loss_lang.data),
        }

        if not model.training:
            # szj 计算语种分类acc
            with torch.no_grad():
                correct_lang = self.get_lang_correct(net_output['language_logits'], language_id)
                logging_output['correct_lang'] = correct_lang

            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    # szj 只更新lid 模块
    def forward_lid(self, model, sample, net_output, language_id, reduce=True):
        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )
        
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens

        # szj 将句子级别的lang loss归到token级别
        # sample size, nsentences 在这套代码中都表示batch中的句子数
        loss_lang = self.get_language_class_loss(net_output['language_logits'], language_id)  * (ntokens / sample["id"].numel()) 
        loss = loss_lang

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "loss_lang": utils.item(loss_lang.data),
        }

        if not model.training:
            # szj 计算语种分类acc
            with torch.no_grad():
                correct_lang = self.get_lang_correct(net_output['language_logits'], language_id)
                logging_output['correct_lang'] = correct_lang

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        # szj
        loss_ctc_sum = utils.item(sum(log.get("loss_ctc", 0) for log in logging_outputs))
        loss_lang_sum = utils.item(sum(log.get("loss_lang", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        # szj
        metrics.log_scalar(
            "loss_ctc", loss_ctc_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_lang", loss_lang_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            # szj
            metrics.log_scalar(
                "nll_loss_ctc", loss_ctc_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_scalar(
                "nll_loss_lang", loss_lang_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        # if c_total > 0:
        metrics.log_derived(
            "uer",
            lambda meters: safe_round(
                meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
            )
            if meters["_c_total"].sum > 0
            else float("nan"),
        )
        # if w_total > 0:
        metrics.log_derived(
            "wer",
            lambda meters: safe_round(
                meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
            )
            if meters["_w_total"].sum > 0
            else float("nan"),
        )
        metrics.log_derived(
            "raw_wer",
            lambda meters: safe_round(
                meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
            )
            if meters["_w_total"].sum > 0
            else float("nan"),
        )
        
        # szj
        correct_lang_total = sum(log.get("correct_lang", 0) for log in logging_outputs)
        metrics.log_scalar("_correct_lang_total", correct_lang_total)
        metrics.log_scalar("_sample_size", sample_size)
        
        if correct_lang_total > 0:
            metrics.log_derived(
                "accuracy_language",
                lambda meters: safe_round(
                    meters["_correct_lang_total"].sum * 100.0 / meters["_sample_size"].sum, 3
                )
            )
            metrics.log_derived(
                "correct_language",
                lambda meters: safe_round(
                    meters["_correct_lang_total"].sum / 1, 3
                )
            )
            metrics.log_derived(
                "all_language",
                lambda meters: safe_round(
                    meters["_sample_size"].sum / 1, 3
                )
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
