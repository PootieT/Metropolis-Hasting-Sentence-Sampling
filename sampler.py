from enum import Enum
from typing import *
import random

import numpy as np
import scipy.special
import torch
import torch.nn as nn
from scipy.stats import poisson, uniform
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

from proposal_acceptor import ProposalAcceptor

COSINE = lambda x, y: (1 - nn.CosineSimilarity(dim=-1)(x.unsqueeze(1), y.unsqueeze(0)))


class SentenceSampler:
    def __init__(self, generator_model_name:str):
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(generator_model_name)

    def get_random_word_covariance(self, input_ids: Tensor, attention_mask: Tensor):
        n = input_ids[0].shape[0]
        C = torch.zeros(n, n)
        ind = np.diag_indices(n)
        C[ind[0], ind[1]] = torch.randn(n)
        return C

    def get_static_word_covariance(
        self,
        input_ids: Tensor,
        attention_mask: Tensor
    ):
        static_embs = self.model.bert.embeddings(input_ids).squeeze(0)  # assume input is one sentence at a time
        cos_sim = 1 - COSINE(static_embs, static_embs)

        start = 1 if input_ids[0][0] == self.tokenizer.bos_token_id else 0
        end = -1 if input_ids[0][-1] == self.tokenizer.eos_token_id else len(input_ids)
        return cos_sim[start:end, start:end]

    def get_weighted_static_contextual_word_covariance(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        lambda_static: float = 0.5,
    ) -> Tensor:
        static_embs = self.model.bert.embeddings(input_ids).squeeze(0)
        contextual_embs = (
            self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].squeeze(0)
        )
        static_cos_sim = 1 - COSINE(static_embs, static_embs)
        contextual_cos_sim = 1 - COSINE(contextual_embs, contextual_embs)
        cos_sim = lambda_static * static_cos_sim + (1 - lambda_static) * contextual_cos_sim

        start = 1 if input_ids[0][0] == 101 else 0
        end = -1 if input_ids[0][-1] == 102 else len(input_ids)
        return cos_sim[start:end, start:end]

    def get_one_pass_perturbed_masking_word_covariance(
        self,
        input_ids: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """ mask each word, calculate change in rest of words' hidden state """
        n = input_ids[0].shape[0]
        start = 1 if input_ids[0][0] == 101 else 0
        end = len(input_ids[0])-1 if input_ids[0][-1] == 102 else len(input_ids[0])

        contextual_embs = (
            self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].squeeze(0)
        )
        covariance = torch.ones(n, n)
        for i in range(start, end):
            masked_input_ids = input_ids.clone()
            masked_input_ids[0][i] = self.tokenizer.mask_token_id
            masked_contextual_embs = (
                self.model(masked_input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].squeeze(0)
            )
            # larger impact a masked token has on rest of the token, the smaller the similarity
            covariance[i, :] = 1 - nn.CosineSimilarity(dim=-1)(contextual_embs, masked_contextual_embs)


        ind = np.diag_indices(covariance.shape[0])
        covariance[ind[0], ind[1]] = 1
        return covariance[start:end, start:end]

    def get_perturbed_masking_word_covariance(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        adjacent_only: bool=False
    ) -> Tensor:
        n = input_ids[0].shape[0]
        start = 1 if input_ids[0][0] == 101 else 0
        end = len(input_ids[0]) - 1 if input_ids[0][-1] == 102 else len(input_ids[0])

        covariance = torch.ones(n, n)
        for i in range(start, end):
            masked_input_ids = input_ids.clone()
            masked_input_ids[0][i] = self.tokenizer.mask_token_id
            masked_contextual_embs = self.model(masked_input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].squeeze(0)
            j_range = [i-1, i+1] if adjacent_only else range(start, end)
            for j in j_range:
                if j != i:
                    masked_2_input_ids = masked_input_ids.clone()
                    masked_2_input_ids[0][j] = self.tokenizer.mask_token_id
                    masked_2_contextual_embs = self.model(masked_2_input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].squeeze(0)
                    covariance[i, j] = 1 - nn.CosineSimilarity(dim=-1)(masked_contextual_embs[i], masked_2_contextual_embs[i])

            if adjacent_only:  # sum diagonals as average of I(i, j-1) and I(i, j+1)
                covariance[i, i] = (covariance[i, i-1] + covariance[i, i+1]) / 2

        return covariance[start:end, start:end]

    def calculate_higher_order_span_score(
        self,
        C: Tensor,
        single_word_logit_penalty: float = 1.0,
        higher_order_penalty: float = 0.8
    ) -> Tensor:
        n = C.shape[0]

        C = (C + C.T) / 2  # make it symmetrical if input is not strictly "covariance matrix"

        # set diagonal values for single word sampling
        single_word_penalty = torch.ones_like(C)
        single_word_penalty -= 1 - torch.eye(n) * single_word_logit_penalty
        C = C * single_word_penalty

        # dynamic algorithm to calculate average score of a span by averaging scores of all its sub-spans
        if n > 2:
            higher_order_ratio = torch.ones_like(C)
            for i in range(2, n):
                for j in range(n - 2, -1, -1):
                    sub_span_coef = (i - 1 - j) * (i - j) / 2
                    C[i, j] = C[i, j] + (C[i - 1, j] + C[i, j + 1]) * sub_span_coef
                    C[i, j] /= sub_span_coef * 2 + 1

                    # fill up higher order diagonal multiplier
                    higher_order_ratio = higher_order_penalty ** (i - 1 - j)

            # apply higher order span penalty
            C = C * higher_order_ratio
        return C

    def get_token_covariance(
        self,
        input_ids: List[int],
        attention_mask: List[int],
        method: str="static",
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
        :return:
        """
        covariance_func = {
            "random": self.get_random_word_covariance,
            "static": self.get_static_word_covariance,
            "mixed": self.get_weighted_static_contextual_word_covariance,
            "mask_one": self.get_one_pass_perturbed_masking_word_covariance,
            "mask_two": self.get_perturbed_masking_word_covariance,
            "mask_two_adjacent": self.get_perturbed_masking_word_covariance
        }
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if method == "mask_two_adjacent":
            kwargs["adjacent_only"] = True
        C = covariance_func["method"](**kwargs)
        return C

    def get_masked_sentence(
        self,
        input_ids: List[int],
        covariance: Tensor,
        sample_span: bool = True,
        action: Optional[str]=None,
        replacement_lambda: Optional[float]=None,
    ) -> List[int]:
        """
        :param input_ids: input sentence tokens
        :param covariance: word covariance/ dependence matrix
        :param sample_span: whether to sample only words or spans as well
        :param action: One of [sub, ins, del], if None, assumes using replacement lambda to sample a target mask len
        :param replacement_lambda: Sample the target mask len from a poisson distribution given lambda. The samller the
            lambda, more likely the sampled sentence will be shorter than original sentence, vice versa.
        :return: masked sentence
        """
        assert (action is None) or (replacement_lambda is None), "either use discrete action or sampling"
        n = covariance.shape[0]
        mask_token_id = self.tokenizer.mask_token_id
        if sample_span:
            indices = torch.tril_indices(n, n)
            p_spans = torch.softmax(covariance[indices], dim=0)
            span_idx = p_spans.multinomial(num_samples=1, replacement=True)
            i = np.floor(np.sqrt(0.25+2*span_idx) - 0.5)
            triangular_num = i*(i+1)/2
            j = span_idx - triangular_num
            mask_indices = list(range(i, j))
        else:
            # sample only from diagonals
            p_tokens = torch.softmax(torch.diag(covariance), dim=0)
            mask_indices = p_tokens.multinomial(num_samples=1, replacement=True)

        mask_indices = np.array(mask_indices) + 1 if input_ids[0] == 101 else np.array(mask_indices)

        if action is not None:
            # TODO be careful of bos/eos index
            if action == "sub":
                input_ids[mask_indices[0]:mask_indices[-1]+2] = [mask_token_id]*len(mask_indices)
            elif action == "del":
                input_ids = input_ids[:mask_indices[0]] + input_ids[mask_indices[-1]:]
            elif action == "ins":
                input_ids[mask_indices[0]:mask_indices[-1] + 2] = [mask_token_id] * len(mask_indices)
                if random.randint(0,1):
                    input_ids.insert(mask_indices[0], mask_token_id)
                else:
                    input_ids.insert(mask_indices[-1]+len(mask_indices), mask_token_id)
            else:
                raise NotImplementedError
        else:  # sampling
            sampled_len = poisson.rvs(scale=replacement_lambda)
            input_ids = input_ids[:mask_indices[0]] + [self.tokenizer.mask_token_id]*sampled_len +input_ids[mask_indices[-1]+1:]
        return input_ids

    def sample_masked_sentence(
        self,
        input_ids:List[str],
        attention_mask: List[str],
        temperature: float=1.0
    ) -> str:
        token_logits = model(input_ids, attention_mask=attention_mask).logits
        # Find the location of [MASK] and extract its logits
        mask_token_index = np.argwhere(input_ids == self.tokenizer.mask_token_id)[0, 1]
        mask_token_logits = token_logits[0, mask_token_index, :]
        mask_token_logits = torch.softmax(mask_token_logits / temperature, dim=1)
        for token_idx in range(mask_token_index):
            input_ids[token_idx] = np.random.choice(len(self.tokenizer.vocab), 1, p=mask_token_logits[token_idx])
        return self.tokenizer.decode(input_ids)

    def sample(
        self,
        sentence: str,
        covariance_method: str="random",
        sample_span: bool=False,
        sample_action: Optional[str]=None,
        replacement_lambda: Optional[float]=None,
        temperature: float=1.0
    ) -> str:
        inputs = self.tokenizer(sentence, return_tensors="pt")
        covariance = self.get_token_covariance(inputs["input_ids"], inputs["attention_mask"], covariance_method)
        masked_sentence = self.get_masked_sentence(inputs["input_ids"], covariance, sample_span, sample_action, replacement_lambda)
        sampled_sentence = self.sample_masked_sentence(masked_sentence, inputs["attention_mask"], temperature)
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
        method: str="random",
        lambda_semantic: float=0.5
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
            - span_mask_two: sample span with double pass perturbed masking covariance, sample target mask with poisson
                distribution
        :param lambda_semantic: proportion of importance given to semantics during acceptance calculation (vs. fluency)
        """
        self.sampler = SentenceSampler(sampler_model_name)
        self.acceptor = ProposalAcceptor(acceptor_semantic_model_name, acceptor_fluency_model_name, lambda_semantic)
        self.method = method
        self.sample_kwargs = {}
        if method.startswith("word"):
            self.sample_kwargs["sample_span"] = False
        else:
            self.sample_kwargs["sample_span"] = True
        if method == "word_pm":
            self.sample_kwargs["covariance_method"] = "mask_two_adjacent"
        else:
            self.sample_kwargs["covariance_method"] = "_".join(method.split("_")[1:])
        self.ACTIONS = ["sub", "ins", "del"]

    def sample_action(self, target_sentence, cur_sentence) -> Tuple[Optional[str], Optional[float]]:
        if self.method.startswith("word"):
            action = random.choice(self.ACTIONS)
            replacement_lbd = None
        else:
            action = random.choice(self.ACTIONS)
            replacement_lbd = len(target_sentence.split()) / len(cur_sentence.split())
        return action, replacement_lbd

    def metropolis_hasting_sample(
        self,
        source_sentence:str,
        target_sentence:str,
        steps: int=100,
        annealing_rate: float = 0.01,
        min_temp: float = 0.1
    ) -> List[Dict]:
        cur_sentence = source_sentence
        temp = 1.0
        action, replacement_lbd = self.sample_action(target_sentence, source_sentence)
        self.acceptor.get_acceptance(cur_sentence)
        results = []
        with torch.no_grad():
            for i in tqdm(range(steps)):
                # sample sentence
                self.sample_kwargs.update({"sample_action": action, "replacement_lambda": replacement_lbd})
                proposal_sentence = self.sampler.sample(cur_sentence, **self.sample_kwargs)
                # calculate acceptance based on semantic and fluency score
                sem_logprob, logprob, acceptance = self.acceptor.get_acceptance(proposal_sentence)
                # accept sentence or not
                if uniform.rvs(size=1) <= acceptance:
                    self.acceptor.cur_sem_logprob = sem_logprob
                    self.acceptor.cur_logprob = logprob
                    cur_sentence = proposal_sentence
                # update sampling hyperparameters
                temp=np.maximum(temp*np.exp(-annealing_rate*i), min_temp)  # follow Gumbel-Softmax Annealing schedule https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
                action, replacement_lbd = self.sample_action(target_sentence, cur_sentence)
                # record metrics and results
                results.append({"sentences": cur_sentence, "proposal_sentence": proposal_sentence,
                                "sem_logprob":sem_logprob, "logprob": logprob, "acceptance": acceptance,
                                "action":action,"replacement_lbd": replacement_lbd, "temperature": temp})
        return results


if __name__ == "__main__":
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    # model = AutoModelForMaskedLM.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    # sentence = "I love New York"
    # encoded_input = tokenizer(sentence, return_tensors="pt")
    # with torch.no_grad():
    #     # c1 = get_static_word_covariance(
    #     #     encoded_input["input_ids"],
    #     #     encoded_input["attention_mask"],
    #     #     model
    #     # )
    #     # c2 = get_weighted_static_contextual_word_covariance(
    #     #     encoded_input["input_ids"],
    #     #     encoded_input["attention_mask"],
    #     #     model
    #     # )
    #     # c3 = get_one_pass_perturbed_masking_word_covariance(
    #     #     encoded_input["input_ids"], encoded_input["attention_mask"], model, tokenizer
    #     # )
    #     c4 = get_perturbed_masking_word_covariance(
    #         encoded_input["input_ids"], encoded_input["attention_mask"], model, tokenizer, adjacent_only=True
    #     )
    mhss = MetropolisHastingSentenceSampler(
        sampler_model_name="distilBERT",
        acceptor_semantic_model_name="sentence-transformers/all-MiniLM-L6-v2",
        acceptor_fluency_model_name="distilgpt2",
        method="random",
        lambda_semantic=0.5
    )
    mhss.metropolis_hasting_sample(
        source_sentence="I love New York",
        target_sentence="your cat looks ugly",
        steps=100,
    )
    pass
