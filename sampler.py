from enum import Enum
from typing import *
import random

import numpy as np
import pandas as pd
import scipy.special
import torch
import torch.nn as nn
from scipy.stats import poisson, uniform, dirichlet
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

from proposal_acceptor import ProposalAcceptor

COSINE = lambda x, y: (1 - nn.CosineSimilarity(dim=-1)(x.unsqueeze(1), y.unsqueeze(0)))


class SentenceSampler:
    def __init__(self, generator_model_name: str, cuda: bool=False):
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(generator_model_name)
        self.device = torch.device(torch.cuda.current_device()) if cuda else torch.device("cpu")

    def get_random_word_covariance(self, input_ids: Tensor, attention_mask: Tensor):
        n = input_ids[0].shape[0]
        C = torch.zeros(n, n, device=self.device)
        ind = np.diag_indices(n)
        C[ind[0], ind[1]] = torch.randn(n, device=self.device)
        start = 1 if input_ids[0][0] == self.tokenizer.cls_token_id else 0
        end = -1 if input_ids[0][-1] == self.tokenizer.sep_token_id else len(input_ids)
        return C[start:end, start:end]

    def get_static_word_covariance(self, input_ids: Tensor, attention_mask: Tensor):
        static_embs = self.model.base_model.embeddings(input_ids).squeeze(0)  # assume input is one sentence at a time
        cos_sim = 1 - COSINE(static_embs, static_embs)

        start = 1 if input_ids[0][0] == self.tokenizer.cls_token_id else 0
        end = -1 if input_ids[0][-1] == self.tokenizer.sep_token_id else len(input_ids)
        return cos_sim[start:end, start:end]

    def get_weighted_static_contextual_word_covariance(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        lambda_static: float = 0.5,
    ) -> Tensor:
        static_embs = self.model.base_model.embeddings(input_ids).squeeze(0)
        contextual_embs = (
            self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].squeeze(0)
        )
        static_cos_sim = 1 - COSINE(static_embs, static_embs)
        contextual_cos_sim = 1 - COSINE(contextual_embs, contextual_embs)
        cos_sim = lambda_static * static_cos_sim + (1 - lambda_static) * contextual_cos_sim

        start = 1 if input_ids[0][0] == self.tokenizer.cls_token_id else 0
        end = -1 if input_ids[0][-1] == self.tokenizer.sep_token_id else len(input_ids)
        return cos_sim[start:end, start:end]

    def get_one_pass_perturbed_masking_word_covariance(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """mask each word, calculate change in rest of words' hidden state"""
        n = input_ids[0].shape[0]
        start = 1 if input_ids[0][0] == self.tokenizer.cls_token_id else 0
        end = len(input_ids[0]) - 1 if input_ids[0][-1] == self.tokenizer.sep_token_id else len(input_ids[0])
        bs = 8

        contextual_embs = (
            self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].squeeze(0)
        )
        covariance = torch.ones(n, n, device=self.device)
        # for i in range(start, end):
        #     # TODO parallelize this
        #     masked_input_ids = input_ids.clone()
        #     masked_input_ids[0][i] = self.tokenizer.mask_token_id
        #     masked_contextual_embs = (
        #         self.model(masked_input_ids, attention_mask=attention_mask, output_hidden_states=True)
        #         .hidden_states[-1]
        #         .squeeze(0)
        #     )
        #     # larger impact a masked token has on rest of the token, the smaller the similarity
        #     covariance[i, :] = 1 - nn.CosineSimilarity(dim=-1)(contextual_embs, masked_contextual_embs)

        input_ids_batch = input_ids.repeat([end-start, 1])
        attention_mask_batch = attention_mask.repeat([end - start, 1])
        input_ids_batch[range(start-1, end-1), range(start, end)] = self.tokenizer.mask_token_id
        for i in range((end-start-1)//bs + 1):
            masked_contextual_embs = self.model(
                input_ids_batch[i*bs: i*bs+bs], attention_mask=attention_mask_batch[i*bs: i*bs+bs], output_hidden_states=True).hidden_states[-1]
            covariance[i*bs+start: i*bs+start+masked_contextual_embs.shape[0], :] = 1 - nn.CosineSimilarity(dim=-1)(contextual_embs, masked_contextual_embs)

        ind = np.diag_indices(covariance.shape[0])
        covariance[ind[0], ind[1]] = 1
        return covariance[start:end, start:end]

    def get_perturbed_masking_word_covariance(
        self, input_ids: Tensor, attention_mask: Tensor, adjacent_only: bool = False
    ) -> Tensor:
        n = input_ids[0].shape[0]
        start = 1 if input_ids[0][0] == self.tokenizer.cls_token_id else 0
        end = len(input_ids[0]) - 1 if input_ids[0][-1] == self.tokenizer.sep_token_id else len(input_ids[0])
        bs = 8

        covariance = torch.ones(n, n, device=self.device)
        # for i in range(start, end):
        #     # TODO parallelize this
        #     masked_input_ids = input_ids.clone()
        #     masked_input_ids[0][i] = self.tokenizer.mask_token_id
        #     masked_contextual_embs = (
        #         self.model(masked_input_ids, attention_mask=attention_mask, output_hidden_states=True)
        #         .hidden_states[-1]
        #         .squeeze(0)
        #     )
        #     j_range = [i - 1, i + 1] if adjacent_only else range(start, end)
        #     for j in j_range:
        #         if j != i:
        #             masked_2_input_ids = masked_input_ids.clone()
        #             masked_2_input_ids[0][j] = self.tokenizer.mask_token_id
        #             masked_2_contextual_embs = (
        #                 self.model(masked_2_input_ids, attention_mask=attention_mask, output_hidden_states=True)
        #                 .hidden_states[-1]
        #                 .squeeze(0)
        #             )
        #             covariance[i, j] = 1 - nn.CosineSimilarity(dim=-1)(
        #                 masked_contextual_embs[i], masked_2_contextual_embs[i]
        #             )
        #
        #     if adjacent_only:  # sum diagonals as average of I(i, j-1) and I(i, j+1)
        #         covariance[i, i] = (covariance[i, i - 1] + covariance[i, i + 1]) / 2

        # parallel version of adjacent only
        input_ids_batch = input_ids.repeat([end - start, 1])
        attention_mask_batch = attention_mask.repeat([end - start, 1])
        input_ids_batch[range(start - 1, end - 1), range(start, end)] = self.tokenizer.mask_token_id
        mask1_embs = []
        for i in range((end - start - 1) // bs + 1):
            masked_contextual_embs = self.model(
                input_ids_batch[i * bs: i * bs + bs], attention_mask=attention_mask_batch[i * bs: i * bs + bs],
                output_hidden_states=True).hidden_states[-1]
            mask1_embs.append(masked_contextual_embs)
        mask1_embs = torch.cat(mask1_embs, dim=0)

        input_ids_batch = input_ids.repeat([(end - start) * 2, 1])
        attention_mask_batch = attention_mask.repeat([(end - start) * 2, 1])
        # in addition to mask the word, we mask word before in one example, and the word after in the second example
        input_ids_batch[
            range(0, (end - start) * 2), np.array([[i]*2 for i in range(start, end)]).reshape(-1).tolist()
        ] = self.tokenizer.mask_token_id
        input_ids_batch[range(0, (end - start) * 2, 2), range(start-1, end-1)] = self.tokenizer.mask_token_id
        input_ids_batch[range(1, (end - start) * 2, 2), range(start + 1, end +1)] = self.tokenizer.mask_token_id
        mask2_embs = []
        for i in range(((end - start) * 2 - 1) // bs + 1):
            masked_contextual_embs = self.model(
                input_ids_batch[i * bs: i * bs + bs], attention_mask=attention_mask_batch[i * bs: i * bs + bs],
                output_hidden_states=True).hidden_states[-1]
            mask2_embs.append(masked_contextual_embs)
        mask2_embs = torch.cat(mask2_embs, dim=0)

        for i in range(0, end-start-1):
            before_after_sim = 1 - nn.CosineSimilarity(dim=-1)(
                mask1_embs[i,i+start], mask2_embs[i*2:i*2+2, i+start]
            )
            covariance[i+start, [i+start - 1, i+start + 1]] = before_after_sim
            covariance[i+start, i+start] = before_after_sim.mean()

        return covariance[start:end, start:end]

    def calculate_higher_order_span_score(
        self, C: Tensor, single_word_logit_penalty: float = 1.0, higher_order_penalty: float = 0.8
    ) -> Tensor:
        n = C.shape[0]

        C = (C + C.T) / 2  # make it symmetrical if input is not strictly "covariance matrix"

        # set diagonal values for single word sampling
        single_word_penalty = torch.ones_like(C, device=self.device)
        single_word_penalty -= 1 - torch.eye(n, device=self.device) * single_word_logit_penalty
        C = C * single_word_penalty

        # dynamic algorithm to calculate average score of a span by averaging scores of all its sub-spans
        if n > 2:
            higher_order_ratio = torch.ones_like(C, device=self.device)
            for i in range(1, n):  # row, up to down
                for j in range(i - 1, -1, -1):   # col, right to left, fill up lower-triangular entries
                    # i-j is how many rows is the span off from diagonal, aka span length
                    off_diag = i-j
                    sub_span_coef = (off_diag) * (off_diag + 1) / 2
                    C[i, j] = C[i, j] + (C[i - 1, j] + C[i, j + 1]) * sub_span_coef
                    if (off_diag)>=2:  # if it's more than 1 diagonal offset, we have to subtract overlapping region
                        sub_sub_span_coef = (off_diag - 1) * (off_diag) / 2
                        C[i, j] -= C[i-1, j+1] * sub_sub_span_coef
                    C[i, j] /= (off_diag+1) * (off_diag + 2) / 2

                    # fill up higher order diagonal multiplier
                    higher_order_ratio[i, j] = higher_order_penalty ** (off_diag)

            # apply higher order span penalty
            C = C * higher_order_ratio
        return C

    def get_token_covariance(
        self,
        input_ids: List[int],
        attention_mask: List[int],
        method: str = "static",
    ) -> Tensor:
        """
        factory method to obtain covariance between words
        :param input_ids:
        :param attention_mask:
        :param method: one of {random, static, mixed, mask_one, mask_two, mask_two_adjacent}
            random - randomly mask out one word O(1)
            mask_two_adjacent - perturbed masking O(2n)
            static - static word embedding covariance O(1)
            mixed - weighted average between static and dynamic word embedding covariance O(1)*
            mask_one - mask one token, calculate perturbance on other tokens O(n)
            mask_two - mask two token at a time calculate i,j dependence O(n*2)
        :return: covariance matrix of shape n X n, where n is number of tokens in a sentence (without
            [CLS] and [SEP])
        """
        covariance_func = {
            "random": self.get_random_word_covariance,
            "static": self.get_static_word_covariance,
            "mixed": self.get_weighted_static_contextual_word_covariance,
            "mask_one": self.get_one_pass_perturbed_masking_word_covariance,
            "mask_two": self.get_perturbed_masking_word_covariance,
            "mask_two_adjacent": self.get_perturbed_masking_word_covariance,
        }
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if method == "mask_two_adjacent":
            kwargs["adjacent_only"] = True
        C = covariance_func[method](**kwargs)
        return C

    def get_masked_sentence(
        self,
        input_ids: List[int],
        covariance: Tensor,
        sample_span: bool = True,
        action: Optional[str] = None,
        replacement_lambda: Optional[float] = None,
    ) -> List[int]:
        """
        :param input_ids: input sentence tokens, 1 X n
        :param covariance: word covariance/ dependence matrix
        :param sample_span: whether to sample only words or spans as well
        :param action: One of [sub, ins, del], if None, assumes using replacement lambda to sample a target mask len
        :param replacement_lambda: Sample the target mask len from a poisson distribution given lambda. The samller the
            lambda, more likely the sampled sentence will be shorter than original sentence, vice versa.
        :return: masked sentence
        """
        assert (action is None) or (replacement_lambda is None), "either use discrete action or sampling"
        input_ids = input_ids.squeeze(0)
        n = covariance.shape[0]
        mask_token_id = self.tokenizer.mask_token_id
        if sample_span:
            indices = torch.tril_indices(n, n)
            p_spans = torch.softmax(covariance[indices[0], indices[1]], dim=0)
            span_idx = p_spans.multinomial(num_samples=1, replacement=True)
            i = torch.floor(torch.sqrt(0.25 + 2 * span_idx) - 0.5)
            triangular_num = i * (i + 1) / 2
            j = span_idx - triangular_num
            mask_span = torch.cat([min(i, j), max(i, j)]).long()
        else:
            # sample only from diagonals
            p_tokens = torch.softmax(torch.diag(covariance), dim=0)
            mask_span = torch.tile(torch.multinomial(p_tokens, 1, replacement=True), [2])

        mask_span = np.array(mask_span.cpu()) + 1 if input_ids[0] == self.tokenizer.cls_token_id else np.array(mask_span)

        if action is not None:
            mask_len = mask_span[1] - mask_span[0] + 1
            if action == "sub":
                input_ids[mask_span[0] : mask_span[1] + 1] = torch.tensor([mask_token_id] * mask_len, device=self.device)
            elif action == "del":
                input_ids = torch.cat([input_ids[: mask_span[0]], input_ids[mask_span[1] + 1 :]])
            elif action == "ins":
                input_ids = torch.cat(
                    [
                        input_ids[: mask_span[0]],
                        torch.tensor([mask_token_id] * (mask_len + 1), device=self.device),
                        input_ids[mask_span[1] + 1 :],
                    ]
                )
            else:
                raise NotImplementedError
        else:  # sampling
            sampled_len = poisson.rvs(replacement_lambda)
            input_ids = torch.cat([
                            input_ids[: mask_span[0]],
                            torch.tensor([mask_token_id] * sampled_len, device=self.device),
                            input_ids[mask_span[-1] + 1 :]
                        ]).long()
        return input_ids.unsqueeze(0)

    def sample_masked_sentence(
        self,
        input_ids: List[str],
        temperature: float = 1.0,
        tgt_embs: Optional[Tensor] = None,
        tgt_hiddens: Optional[Tensor] = None,
        tgt_logits: Optional[Tensor] = None,
        fusion_aggregation: str = "closest",
        fusion_interpolation: str = "linear",
    ) -> str:
        assert not (tgt_embs is not None and tgt_logits is not None), "only one type of fusion at a time"
        mask_token_index = np.argwhere((input_ids == self.tokenizer.mask_token_id).cpu())[1]
        if len(mask_token_index) == 0:
            return self.tokenizer.decode(input_ids[0][1:-1])

        mask_span = [min(mask_token_index), max(mask_token_index)]

        if tgt_embs is not None:
            input_embs = self.model.base_model.embeddings(input_ids)
            input_embs = self.interpolate_states(
                input_embs, tgt_embs, mask_span, fusion_aggregation, fusion_interpolation
            )
            token_logits = self.model(
                inputs_embeds=input_embs,
                attention_mask=torch.ones_like(input_ids, device=self.device)
            ).logits
        elif tgt_hiddens is not None:
            token_hiddens = self.model(
                input_ids,
                attention_mask=torch.ones_like(input_ids, device=self.device),
                output_hidden_states=True
            ).hidden_states[-1]
            token_hiddens = self.interpolate_states(
                token_hiddens, tgt_hiddens, mask_span, fusion_aggregation, fusion_interpolation
            )
            # https://github.com/huggingface/transformers/blob/31d452c68b34c2567b62924ee0df40a83cbc52d5/src/transformers/models/distilbert/modeling_distilbert.py#L673
            token_logits = self.model.vocab_transform(token_hiddens)  # (bs, seq_length, dim)
            token_logits = self.model.activation(token_logits)        # (bs, seq_length, dim)
            token_logits = self.model.vocab_layer_norm(token_logits)  # (bs, seq_length, dim)
            token_logits = self.model.vocab_projector(token_logits)   # (bs, seq_length, vocab_size)
        else:
            token_logits = self.model(
                input_ids,
                attention_mask=torch.ones_like(input_ids, device=self.device),
            ).logits

        if tgt_logits is not None:
            token_logits = self.interpolate_states(
                token_logits, tgt_logits, mask_span, fusion_aggregation, fusion_interpolation
            )

        mask_token_logits = token_logits[0, mask_token_index, :]
        mask_token_logits = torch.softmax(mask_token_logits / temperature, dim=1)
        input_ids[0, mask_token_index] = torch.multinomial(mask_token_logits, 1, replacement=True).squeeze(1)
        return self.tokenizer.decode(input_ids[0][1:-1])

    def interpolate_states(
        self,
        src_embs: Tensor,
        tgt_embs: Tensor,
        mask_span: List[int],
        aggregate_method: str = "local",
        interpolate_method: str = "linear",
    ):
        """
        interpolate between source and target embeddings (contextual or static)
        :param src_embs: 1 x S x E embedding matrix
        :param tgt_embs: 1 x T x E embedding matrix
        :param aggregate_method: one of {closest, local, global}
            closest: round to closest
        :param interpolate_method: one of {linear, exp, polar}
        :return: embedding matrix same shape as src_emb
        """
        src_len = src_embs.shape[1] - 2
        tgt_len = tgt_embs.shape[1] - 2
        new_src_embs = src_embs.clone()
        for src_idx in range(mask_span[0], mask_span[1] + 1):
            tgt_idx = src_idx / src_len * tgt_len
            # TODO aggregation could also just sample an index
            if aggregate_method == "closest":
                tgt_emb = tgt_embs[0, round(tgt_idx)]
            elif aggregate_method == "local":
                tgt_idx_int = int(np.floor(tgt_idx))
                frac = tgt_idx - tgt_idx_int
                tgt_emb = self.interpolate(
                    embs=tgt_embs[0, tgt_idx_int: tgt_idx_int + 2],
                    method=interpolate_method,
                    weights=torch.tensor([1 - frac, frac], device=self.device),
                )
            elif aggregate_method.startswith("local_"):
                window_size = int(np.floor(int(aggregate_method.replace("local_",""))/2))
                tgt_idx_int = int(np.floor(tgt_idx))
                left, right = max(1, tgt_idx_int-window_size+1), min(tgt_idx_int + window_size + 2, tgt_len)
                alpha = torch.tensor([np.abs(i - tgt_idx) for i in range(left, right)], device=self.device)
                alpha = (-alpha)+alpha.max()+1
                tgt_emb = self.interpolate(
                    embs=tgt_embs[0, left: right],
                    method=interpolate_method,
                    weights=alpha,
                    sample_weight=True
                )
            elif aggregate_method == "global":
                alpha = torch.tensor([np.abs(tgt_len- i + tgt_idx-1) for i in range(tgt_len)], device=self.device)
                tgt_emb = self.interpolate(
                    embs=tgt_embs[0, 1:-1], method=interpolate_method, weights=alpha, sample_weight=True)
            else:
                raise NotImplementedError(f"aggregation method={aggregate_method} not implemented")
            mix_emb = self.interpolate(
                embs=torch.stack([src_embs[0, src_idx], tgt_emb]),
                method=interpolate_method,
                weights=torch.tensor([0.5, 0.5], device=self.device),
                sample_weight=False
            )
            new_src_embs[0, src_idx] = mix_emb
        return new_src_embs

    def interpolate(self, embs: Tensor, method: str = "linear", weights: Optional[Tensor] = None, sample_weight: bool=False):
        if weights is None:
            weights = torch.ones(len(embs), device=self.device) / len(embs)

        if method == "polar" and sample_weight:
            # polar or dirchlet polar: https://aclanthology.org/2022.aacl-short.50.pdf
            # here we treat weights as concentration alpha
            weights = torch.tensor(dirichlet.rvs(weights, size=1), device=self.device).squeeze(0)
        elif method in ["linear", "exp"] and abs(weights.sum()-1) >= 1e-3:
            weights = weights / weights.sum()

        assert abs(weights.sum()-1) < 1e-3
        if method == "linear":
            return torch.matmul(weights.float(), embs)
        elif method == "exp":
            # return torch.prod(torch.pow(embs, weights.repeat(embs.shape[-1],1).T), dim=0)  # numerically unstable
            return torch.exp(torch.matmul(weights.float(), embs))
        elif method == "polar":
            return torch.matmul(torch.sqrt(weights).float(), embs)
        else:
            raise NotImplementedError(f"interpolation method={method} not implemented")

    def sample(
        self,
        sentence: str,
        covariance_method: Optional[str] = "random_word",
        sample_span: bool = False,
        single_word_penalty: float = 1.0,
        higher_order_penalty: float = 0.8,
        sample_action: Optional[str] = None,
        replacement_lambda: Optional[float] = None,
        temperature: float = 1.0,
        tgt_embs: Optional[Tensor] = None,
        tgt_hiddens: Optional[Tensor] = None,
        tgt_logits: Optional[Tensor] = None,
        fusion_aggregation: str = "closest",
        fusion_interpolation: str = "linear",
    ) -> str:
        inputs = self.tokenizer(sentence, return_tensors="pt", max_length=self.tokenizer.max_len_single_sentence)
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        if self.model.device != self.device:
            self.model = self.model.to(self.device)

        if covariance_method is not None:
            covariance = self.get_token_covariance(inputs["input_ids"], inputs["attention_mask"], covariance_method)
            if sample_span:
                covariance = self.calculate_higher_order_span_score(covariance, single_word_penalty, higher_order_penalty)
            masked_sentence = self.get_masked_sentence(
                inputs["input_ids"], covariance, sample_span, sample_action, replacement_lambda
            )
        else:
            masked_sentence = inputs["input_ids"]
        sampled_sentence = self.sample_masked_sentence(
            masked_sentence, temperature, tgt_embs, tgt_hiddens, tgt_logits, fusion_aggregation, fusion_interpolation
        )

        # self.model = self.model.to("cpu")
        return sampled_sentence


class MetropolisHastingSentenceSampler:
    """
    Main class for Metropolis Hasting Sentence Sampling algorithm
    """

    def __init__(
        self,
        sampler_model_name: str,
        acceptor_semantic_model_name: str,
        acceptor_fluency_model_name: str,
        method: Optional[str] = "word_random",
        lambda_semantic: float = 0.5,
        lambda_fluency: float=1.0,
        target_fusion: Optional[str] = None,
        fusion_aggregation: str = "closest",
        fusion_interpolation: str = "linear",
        annealing_rate: float = 0.01,
        init_temp: float = 10.0,
        min_temp: float = 2.0,
        cuda: bool=False,
    ):
        """

        :param sampler_model_name: name of huggingface model to use for generating samples
        :param acceptor_semantic_model_name: name of huggingface model to use for evaluating semantic similarity between
            sample and target sentence
        :param acceptor_fluency_model_name: name of huggingface model to use for evaluating fluency
        :param method: one of {word_random, word_pm, span_static, span_mixed, span_mask_one, span_mask_two}
            - word_random: similar to CGMH (https://arxiv.org/pdf/1811.10996.pdf), randomly select word, randomly choose
                one of three actions {delete, insert, substitute}
            - word_pm: similar to perturbed masking (https://aclanthology.org/2022.findings-acl.111.pdf), select word
                based on edit score, randomly choose one of three actions {delete, insert, substitute}
            - span_static: sample span with static covariance, sample target mask with poisson distribution
            - span_mixed: sample span with mix covariance (static & contextual), sample target mask with poisson
                distribution
            - span_mask_one: sample span with one pass perturbed masking covariance, sample target mask with poisson
                distribution
            - span_pm:
            - span_mask_two: sample span with double pass perturbed masking covariance, sample target mask with poisson
                distribution
            - None: probing mode, no sampling
        :param lambda_semantic: exponent scalar to weigh semantic probability difference, higher it is acceptance is
            more significantly influenced by semantic scores
        :param lambda_fluency: exponent scalar to weigh average token probability difference, higher it is acceptance is
            more significantly influenced by fluency scores
        :param target_fusion: one of {None, emb, logit}
            - None: no target fusion, samples are purely from language model based on source
            - embs: fusion at last hidden state between source and target
            - logits: fusion at logit level between source and target
        :param fusion_aggregation: one of {closest, local, global, local_x}, local_x (where x can be any integer) will
            aggregate with neighborhood size of x (x/2 ahead and x/2 behind the word)
        :param fusion_interpolation: one of {linear, polar, exp}
        :param annealing_rate:
        :param init_temp:
        :param min_temp:
        """
        self.sampler = SentenceSampler(sampler_model_name, cuda=cuda)
        self.acceptor = ProposalAcceptor(
            acceptor_semantic_model_name, acceptor_fluency_model_name, lambda_semantic, lambda_fluency, cuda=cuda)
        self.method = method
        self.sample_kwargs = {"fusion_aggregation": fusion_aggregation, "fusion_interpolation": fusion_interpolation}
        if method is None:
            self.sample_kwargs["covariance_method"] = None
        else:
            if method.startswith("word"):
                self.sample_kwargs["sample_span"] = False
            else:
                self.sample_kwargs["sample_span"] = True
            if method in ["word_pm", "span_pm"]:
                self.sample_kwargs["covariance_method"] = "mask_two_adjacent"
            else:
                self.sample_kwargs["covariance_method"] = "_".join(method.split("_")[1:])
        self.ACTIONS = ["sub", "ins", "del"]
        self.target_fusion = target_fusion
        self.annealing_rate = annealing_rate
        self.init_temp = init_temp
        self.min_temp = min_temp
        self.device = torch.device(torch.cuda.current_device()) if cuda else torch.device("cpu")
        self.sampler.device = self.device
        self.sampler.model = self.sampler.model.to(self.device)
        self.acceptor.device = self.device

    def sample_action(self, target_sentence, cur_sentence) -> Tuple[Optional[str], Optional[float]]:
        if self.method.startswith("word"):
            action = random.choice(self.ACTIONS)
            replacement_lbd = None
        else:
            action = None
            replacement_lbd = len(target_sentence.split()) / len(cur_sentence.split())
        return action, replacement_lbd

    def metropolis_hasting_sample(
        self,
        source_sentence: str,
        target_sentence: str,
        steps: int = 100,
        early_stop_semantic_ratio: Optional[float] = None
    ) -> pd.DataFrame:
        cur_sentence = source_sentence
        temp = self.init_temp
        action, replacement_lbd = self.sample_action(target_sentence, source_sentence)
        if self.target_fusion is not None:
            self.prepare_fusion_signals(target_sentence, source_sentence)
        self.acceptor.set_target_sentence(target_sentence)
        self.acceptor.get_acceptance(cur_sentence)
        init_similarity = torch.exp(self.acceptor.cur_sem_logprob).item()
        results = [{
            "sentences": cur_sentence,
            "proposal_sentence": "",
            "sem_logprob": self.acceptor.cur_sem_logprob.item(),
            "logprob": self.acceptor.cur_logprob.item(),
            "sem_sim": torch.exp(self.acceptor.cur_sem_logprob).item(),
            "semantic_progress": 0.0,
            "avg_perplexity": torch.exp(-self.acceptor.cur_logprob).item(),
            "acceptance": 0.0,
            "action": action,
            "replacement_lbd": replacement_lbd,
            "temperature": temp,
        }]
        with torch.no_grad():
            for i in tqdm(range(steps)):
                if len(cur_sentence) == 0:
                    break
                # sample sentence
                self.sample_kwargs.update({"sample_action": action, "replacement_lambda": replacement_lbd})
                proposal_sentence = self.sampler.sample(cur_sentence, **self.sample_kwargs)
                # calculate acceptance based on semantic and fluency score
                sem_logprob, logprob, acceptance = self.acceptor.get_acceptance(proposal_sentence)
                # accept sentence or not
                if uniform.rvs(size=1) <= acceptance.cpu().numpy():
                    self.acceptor.cur_sem_logprob = sem_logprob
                    self.acceptor.cur_logprob = logprob
                    cur_sentence = proposal_sentence
                # update sampling hyperparameters
                # follow Gumbel-Softmax Annealing https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
                temp = np.maximum(
                    temp * np.exp(-self.annealing_rate * i), self.min_temp
                )
                # temp = np.maximum(temp - self.annealing_rate, self.min_temp)
                action, replacement_lbd = self.sample_action(target_sentence, cur_sentence)
                # record metrics and results
                results.append(
                    {
                        "sentences": cur_sentence,
                        "proposal_sentence": proposal_sentence,
                        "sem_logprob": sem_logprob.item(),
                        "logprob": logprob.item(),
                        "sem_sim": torch.exp(sem_logprob).item(),
                        "semantic_progress": (torch.exp(sem_logprob).item()-init_similarity)/(1-init_similarity),
                        "avg_perplexity": torch.exp(-logprob).item(),
                        "acceptance": acceptance.item(),
                        "action": action,
                        "replacement_lbd": replacement_lbd,
                        "temperature": temp,
                    }
                )
                if early_stop_semantic_ratio is not None and results[-1]["semantic_progress"] > early_stop_semantic_ratio:
                    break

        self.acceptor.clear_state()
        df = pd.DataFrame(results)
        return df

    def prepare_fusion_signals(self, target_sentence, source_sentence):
        tgt_inputs = self.sampler.tokenizer(target_sentence, return_tensors="pt")
        tgt_inputs["input_ids"] = tgt_inputs["input_ids"].to(self.device)
        tgt_inputs["attention_mask"] = tgt_inputs["attention_mask"].to(self.device)
        if self.target_fusion == "embs":
            self.sample_kwargs["tgt_embs"] = self.sampler.model.base_model.embeddings(tgt_inputs["input_ids"])
        elif self.target_fusion == "hiddens":
            self.sample_kwargs["tgt_hiddens"] = self.sampler.model(
                output_hidden_states=True, **tgt_inputs
            ).hidden_states[-1].to(self.device)
        elif self.target_fusion == "logits":
            self.sample_kwargs["tgt_logits"] = self.sampler.model(**tgt_inputs).logits.to(self.device)

        if self.method is None:
            # if at probing mode, we are going to swap masked id in target embedding so it matches up with source, so
            # we can directly interpolate the two masked ids, as opposed to other neighboring tokens
            src_inputs = self.sampler.tokenizer(source_sentence, return_tensors="pt")
            tgt_idx = np.argwhere((tgt_inputs["input_ids"] == self.tokenizer.mask_token_id))[1]
            src_idx = np.argwhere((src_inputs["input_ids"] == self.tokenizer.mask_token_id))[1]
            if self.target_fusion == "hiddens":
                self.sample_kwargs["tgt_hiddens"][0, src_idx] = self.sample_kwargs["tgt_hiddens"][0, tgt_idx]
            elif target_sentence == "logits":
                self.sample_kwargs["tgt_logits"][0, src_idx] = self.sample_kwargs["tgt_logits"][0, tgt_idx]

    def post_sample_filter(
        self,
        sample_df: pd.DataFrame,
        target_semantic_ratio: Optional[float]=None,
        max_perplexity: float=1e10,
        remove_degenerates: bool=False,
    ):
        if remove_degenerates:
            sample_df = sample_df.filter(lambda row: len(row["sentences"].split()) > 1)

        target_semantic_ratio = (max(sample_df.sem_sim)-sample_df.sem_sim[0])/max(sample_df.sem_sim) \
            if target_semantic_ratio is None else target_semantic_ratio

        if target_semantic_ratio > max(sample_df.semantic_progress):
            return None
        idx = min(sample_df[sample_df.semantic_progress >= target_semantic_ratio].index)
        sample_df = sample_df.iloc[:idx+1]
        sample_df = sample_df[sample_df.avg_perplexity < max_perplexity]
        return sample_df


if __name__ == "__main__":
    mhss = MetropolisHastingSentenceSampler(
        sampler_model_name="distilbert-base-uncased",
        acceptor_semantic_model_name="sentence-transformers/all-mpnet-base-v2",  # "sentence-transformers/all-MiniLM-L6-v2",
        acceptor_fluency_model_name="distilgpt2",
        method="word_pm",
        lambda_semantic=10.0,
        lambda_fluency=0.1,
        target_fusion="logits",
        fusion_aggregation="closest",
        fusion_interpolation="exp",
        init_temp=1.0,
        annealing_rate=1e-4,
        min_temp=0.1,
        cuda=True
    )
    mhss.metropolis_hasting_sample(
        source_sentence="I love New York",
        target_sentence="your cat looks ugly",
        steps=100,
    )
    pass
