from dataclasses import asdict, dataclass
from typing import List
import os
import json
from tokenizers import Tokenizer
from transformers import BartForConditionalGeneration
import torch
import math
from .asm import AsmAdder, FuncDataclass
from torch.nn.utils.rnn import pad_sequence
from forklift.par_data import DP
from typing import Optional
InferenceDataProcessor = DP


@dataclass
class Config:
    hf_model_path: str
    pairs: List[str]
    beam: int = 5
    nbest: int = 1
    early_stopping: bool = True
    length_penalty: float = 1.0
    min_length: int = 1
    max_new_tokens: int = 2048
    is_exebench_backend = True
    asm_key = 'real'

    def to_dict(self):
        return asdict(self)


class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        try:
            tok = Tokenizer.from_file(os.path.join(self.config.hf_model_path, 'tokenizer.json'))
        except:
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()
            tok = Tokenizer.from_str(fs.open(os.path.join(self.config.hf_model_path, 'tokenizer.json'), 'r').read())

        self.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

        self.model = BartForConditionalGeneration.from_pretrained(
            self.config.hf_model_path
        ).to(self.device).eval()
        self.model = BartForConditionalGeneration.from_pretrained(
            self.config.hf_model_path
        ).to(self.device).eval()
        self._is_exebench_backend = self.config.is_exebench_backend
        self.asm_key = self.config.asm_key
        self.required_asms = self.get_required_asms()
        self.data_processor = InferenceDataProcessor(tokenizer=tok)

    def get_required_asms(self):
        required_asms = set()
        dp = DP()
        for pair in self.config.pairs:
            source_k, target_k, _, _ = dp.get_par_data(row=None, pair=pair, asm_key='angha', fPIC=False)
            source_k = source_k.replace('angha_', '')
            target_k = target_k.replace('angha_', '')
            required_asms.add(source_k)
            required_asms.add(target_k)

        # always add O0 reference
        new_asm_to_add = set()
        for k in required_asms:
            if '_O0' not in k:
                new_asm_to_add.add(k.replace('_O3', '_O0').replace('_Os', '_O0').replace('_Oz', '_O0'))
        required_asms = required_asms.union(new_asm_to_add)
        return list(required_asms)

    def predict_batch(self, rows_pairs):
        tokenized = []
        max_len = []
        for idx, (r, p) in enumerate(rows_pairs):
            tok, len_t = self.data_processor.prepare(r, p, asm_key=self.asm_key, return_target_length=True)
            if len(tok) > self.model.config.max_position_embeddings or len_t > self.model.config.max_position_embeddings:
                max_len.append(True)
            else:
                tokenized.append(torch.tensor(tok))
                max_len.append(False)

        batch = pad_sequence(
            tokenized,
            True,
            self.data_processor.tokenizer.get_vocab()['<pad>']
        ).to(self.device)

        output = self.model.generate(batch, max_new_tokens=self.config.max_new_tokens, num_beams=self.config.beam,
                                     num_return_sequences=self.config.nbest, early_stopping=self.config.early_stopping,
                                     length_penalty=self.config.length_penalty, min_length=self.config.min_length,
                                     )
        res = []
        output = output.view(len(tokenized), self.config.nbest, -1).cpu()
        skip = 0
        for idx in range(len(rows_pairs)):
            if max_len[idx]:
                res.append([''])
                skip += 1
                continue
            idx_output = idx - skip
            hyps = []
            for out in output[idx_output]:
                detokenized = self.data_processor.detokenize(out.tolist())
                hyps.append(detokenized)
            res.append(hyps)
        return res

    @staticmethod
    def get_asm(key, row):
        asm_idx = row['asm']['target'].index(key)
        return row['asm']['code'][asm_idx]
