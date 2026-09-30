import os

import torch
import transformers
from tqdm import tqdm

import iglm
from iglm.model.tokens import *
from iglm.model.utils import *
from iglm.utils.general import exists

project_path = os.path.dirname(os.path.realpath(iglm.__file__))
trained_models_dir = os.path.join(project_path, 'trained_models')

CHECKPOINT_DICT = {
    "IgLM": os.path.join(trained_models_dir, 'IgLM'),
    "IgLM-S": os.path.join(trained_models_dir, 'IgLM-S'),
}
VOCAB_FILE = os.path.join(trained_models_dir, 'vocab.txt')


class IgLM():

    def __init__(self, model_name: str = "IgLM"):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = transformers.GPT2LMHeadModel.from_pretrained(
            CHECKPOINT_DICT[model_name]).to(self.device)
        self.model.eval()

        self.tokenizer = transformers.BertTokenizerFast(vocab_file=VOCAB_FILE,
                                                        do_lower_case=False)

    def _generate(self, starting_tokens, num_to_generate, top_p, temperature):
        decoded_seqs = set()  # Set to remove duplicates
        pbar = tqdm(total=num_to_generate)
        while len(decoded_seqs) < num_to_generate:
            seq = self.model.generate(
                starting_tokens,
                max_length=150,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.cls_token_id,
                forced_eos_token_id=self.tokenizer.cls_token_id,
                bad_words_ids=BAD_WORD_IDS,
                do_sample=True,
                top_p=top_p,
                temperature=temperature).detach().cpu().numpy()

            seq = seq[0]  # Squeeze out batch dimension
            if validate_generated_seq(seq, self.tokenizer):
                decoded_tokens = self.tokenizer.decode(
                    iglm_to_infilled(seq, self.tokenizer))
                decoded_seq = ''.join(decoded_tokens).replace(' ', '')
                if decoded_seq not in decoded_seqs:
                    decoded_seqs.add(decoded_seq)
                    pbar.update(1)

        pbar.close()
        return list(decoded_seqs)

    def generate(self,
                 chain_token,
                 species_token,
                 prompt_sequence=None,
                 num_to_generate=1000,
                 top_p=1,
                 temperature=1):
        start_tokens = [chain_token, species_token]

        if exists(prompt_sequence):
            prompt_tokens = list(prompt_sequence)
            start_tokens += prompt_tokens

        start_tokens = torch.Tensor([
            self.tokenizer.convert_tokens_to_ids(start_tokens)
        ]).int().to(self.device)

        assert (start_tokens != self.tokenizer.unk_token_id
                ).all(), "Unrecognized token supplied in starting tokens"

        generated_seqs = self._generate(
            start_tokens,
            num_to_generate=num_to_generate,
            top_p=top_p,
            temperature=temperature,
        )

        return generated_seqs

    def infill(
        self,
        sequence,
        chain_token,
        species_token,
        infill_range,
        num_to_generate=1000,
        top_p=1,
        temperature=1,
    ):
        sequence = list(sequence)
        masked_seq = mask_span(
            sequence,
            infill_range[0],
            infill_range[1],
        )  # mask using provided indices
        start_tokens = [chain_token, species_token] + masked_seq
        start_tokens = torch.Tensor([
            self.tokenizer.convert_tokens_to_ids(start_tokens)
        ]).int().to(self.device)

        assert (start_tokens != self.tokenizer.unk_token_id
                ).all(), "Unrecognized token supplied in starting tokens"

        generated_seqs = self._generate(
            start_tokens,
            num_to_generate=num_to_generate,
            top_p=top_p,
            temperature=temperature,
        )

        return generated_seqs

    def position_probabilities(
        self,
        sequence,
        chain_token,
        species_token,
        alphabet="ACDEFGHIKLMNPQRSTVWY",
        temperature=1.0,
        batch_size=32,
    ):
        """Return a context-conditioned amino-acid distribution per position.

        Each residue is masked independently. IgLM sees both flanks of the
        sequence and predicts the first token of the missing single-residue
        span. The returned matrix has shape ``[len(sequence), len(alphabet)]``.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        sequence = list(sequence)
        amino_acid_ids = self.tokenizer.convert_tokens_to_ids(list(alphabet))
        assert self.tokenizer.unk_token_id not in amino_acid_ids, (
            "Unrecognized amino acid supplied in alphabet"
        )

        contexts = []
        for index in range(len(sequence)):
            tokens = [chain_token, species_token] + mask_span(
                sequence,
                index,
                index + 1,
            )
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            assert self.tokenizer.unk_token_id not in token_ids, (
                "Unrecognized token supplied while building position probabilities"
            )
            contexts.append(token_ids)

        probabilities = []
        with torch.no_grad():
            for start in range(0, len(contexts), batch_size):
                token_batch = torch.tensor(
                    contexts[start:start + batch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                logits = self.model(token_batch).logits[:, -1, amino_acid_ids]
                batch_probabilities = torch.softmax(
                    logits / temperature,
                    dim=-1,
                )
                probabilities.extend(batch_probabilities.detach().cpu().tolist())

        return probabilities

    def log_likelihood(
        self,
        sequence,
        chain_token,
        species_token,
        infill_range=None,
    ):
        sequence = list(sequence)
        if exists(infill_range):
            sequence = mask_span(
                sequence,
                infill_range[0],
                infill_range[1],
                append_span=True,
            )  # mask using provided indices

        token_seq = [chain_token, species_token] + sequence

        if exists(infill_range):
            token_seq += [self.tokenizer.cls_token]
        else:
            token_seq += [self.tokenizer.sep_token]

        token_seq = torch.Tensor([
            self.tokenizer.convert_tokens_to_ids(token_seq)
        ]).int().to(self.device)

        assert (token_seq != self.tokenizer.unk_token_id
                ).all(), "Unrecognized token supplied in starting tokens"

        if exists(infill_range):
            eval_start = np.nonzero(
                token_seq[0] == self.tokenizer.sep_token_id)[0].item()
        else:
            eval_start = 1

        logits = self.model(token_seq).logits
        shift_logits = logits[..., eval_start:-1, :].contiguous()
        shift_labels = token_seq[..., eval_start + 1:].contiguous().long()
        nll = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean',
        )

        return -nll.item()
