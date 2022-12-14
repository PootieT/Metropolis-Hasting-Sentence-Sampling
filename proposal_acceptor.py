from typing import *

import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM


class ProposalAcceptor:
    def __init__(
        self,
        semantic_model_name: Optional[str]=None,
        fluency_model_name: Optional[str]=None,
        lambda_semantic: float=0.5
    ):
        self.semantic_model = SentenceTransformer(semantic_model_name)
        self.ppl_tokenizer = AutoTokenizer.from_pretrained(fluency_model_name)
        self.ppl_lm = AutoModelForCausalLM.from_pretrained(fluency_model_name)
        if self.ppl_tokenizer.pad_token is None:
            self.ppl_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tgt_sem_emb = None
        self.lambda_semantic = lambda_semantic
        self.cur_sem_logprob = None
        self.cur_logprob = None

    def set_target_sentence(self, target_sentence: str):
        self.tgt_sem_emb = self.semantic_model.encode(target_sentence)

    def calculate_semantic_similarity(self, sentence: str) -> Tensor:
        assert self.tgt_sem_emb is not None, "Run set_target_sentence first!"
        emb = self.semantic_model.encode(sentence)
        sem_sim = cosine_similarity(emb.reshape(1,-1), self.tgt_sem_emb.reshape(1,-1))[0][0]
        return torch.tensor(sem_sim)

    def calculate_lm_logprob(self, sentence: List[str], stride: int = 512) -> Tensor:
        # source: https://huggingface.co/docs/transformers/perplexity
        max_length = self.ppl_lm.config.n_positions
        stride = min(max_length, stride)

        encodings = self.ppl_tokenizer(sentence, return_tensors="pt", padding="longest", truncation=True)

        nlls = []
        # if sentences are longer than default window size
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.ppl_lm.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            # padded tokens set to -100, no attention no loss https://github.com/huggingface/transformers/issues/2630
            target_ids[target_ids == self.ppl_tokenizer.vocab[self.ppl_tokenizer.pad_token]] = -100
            # switch it to EOS because model word embedding doesn't have EOS. As long as label is -100 what token it
            # switches to doesn't impact performance
            input_ids[input_ids == self.ppl_tokenizer.vocab[self.ppl_tokenizer.pad_token]] = self.ppl_tokenizer.eos_token_id

            with torch.no_grad():
                # instead of taking aggregated cross entropy from causal LM, we calculate
                # per sentence without reduction.
                # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/gpt2/modeling_gpt2.py#L1072
                outputs = self.ppl_lm(input_ids)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                neg_log_likelihood = loss.view(shift_labels.shape).sum(dim=1)

            nlls.append(neg_log_likelihood)

        sent_lens = encodings.attention_mask.sum(dim=1).to(self.ppl_lm.device)
        # ppl = torch.exp(torch.stack(nlls).sum(dim=0) / sent_lens)
        return -torch.stack(nlls).sum(dim=0) / sent_lens

    def get_acceptance(self, sentence: str) -> Optional[Tuple[Tensor, Tensor, float]]:
        sem_sim = self.calculate_semantic_similarity(sentence)
        sem_logprob = torch.log(sem_sim)
        try:
            logprob = self.calculate_lm_logprob([sentence])[0]
        except Exception as e:
            print(f"logprob calculation error: {e}.\nsentence: {sentence}")
            logprob = torch.tensor(-float("inf"))
        if self.cur_logprob is None:
            self.cur_sem_logprob = sem_logprob
            self.cur_logprob = logprob
            return None
        else:
            accptance = self.lambda_semantic * torch.exp(sem_logprob - self.cur_sem_logprob) + \
                        (1-self.lambda_semantic) * torch.exp(logprob - self.cur_logprob)
            return sem_logprob, logprob, accptance


if __name__ == "__main__":
    acceptor = ProposalAcceptor(
        target_sentence="My cate is cute",
        semantic_model_name='sentence-transformers/all-MiniLM-L6-v2',
        fluency_model_name="distilgpt2"
    )
    s1 = "new york city smells bad"
    s2 = "my dog is a cutie"
    acceptor.get_acceptance(s2)
    accpt1 = acceptor.get_acceptance(s1)
    pass