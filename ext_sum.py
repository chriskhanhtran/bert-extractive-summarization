import time
import numpy as np
import torch
from transformers import BertTokenizer
from nltk.tokenize import sent_tokenize
from models.model_builder import ExtSummarizer


def preprocess(source_fp):
    """
    - Remove \n
    - Sentence Tokenize
    - Add [SEP] [CLS] as sentence boundary
    """
    with open(source_fp) as source:
        raw_text = source.read().replace("\n", " ").replace("[CLS] [SEP]", " ")
    sents = sent_tokenize(raw_text)
    processed_text = "[CLS] [SEP]".join(sents)
    return processed_text, len(sents)


def load_text(processed_text, max_pos, device):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    sep_vid = tokenizer.vocab["[SEP]"]
    cls_vid = tokenizer.vocab["[CLS]"]

    def _process_src(raw):
        raw = raw.strip().lower()
        raw = raw.replace("[cls]", "[CLS]").replace("[sep]", "[SEP]")
        src_subtokens = tokenizer.tokenize(raw)
        src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]
        src_subtoken_idxs = tokenizer.convert_tokens_to_ids(src_subtokens)
        src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
        src_subtoken_idxs[-1] = sep_vid
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        
        segments_ids = []
        segs = segs[:max_pos]
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        mask_src = (1 - (src == 0).float()).to(device)
        cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0
        return src, mask_src, segments_ids, clss, mask_cls

    src, mask_src, segments_ids, clss, mask_cls = _process_src(processed_text)
    segs = torch.tensor(segments_ids)[None, :].to(device)
    src_text = [[sent.replace("[SEP]", "").strip() for sent in processed_text.split("[CLS]")]]
    return src, mask_src, segs, clss, mask_cls, src_text


def test(model, input_data, result_path, max_length, block_trigram=True):
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i : i + n]))
        return ngram_set

    def _block_tri(c, p):
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    with open(result_path, "w") as save_pred:
        with torch.no_grad():
            src, mask, segs, clss, mask_cls, src_str = input_data
            sent_scores, mask = model(src, segs, clss, mask, mask_cls)
            sent_scores = sent_scores + mask.float()
            sent_scores = sent_scores.cpu().data.numpy()
            selected_ids = np.argsort(-sent_scores, 1)

            pred = []
            for i, idx in enumerate(selected_ids):
                _pred = []
                if len(src_str[i]) == 0:
                    continue
                for j in selected_ids[i][: len(src_str[i])]:
                    if j >= len(src_str[i]):
                        continue
                    candidate = src_str[i][j].strip()
                    if block_trigram:
                        if not _block_tri(candidate, _pred):
                            _pred.append(candidate)
                    else:
                        _pred.append(candidate)

                    if len(_pred) == max_length:
                        break

                _pred = " ".join(_pred)
                pred.append(_pred)

            for i in range(len(pred)):
                save_pred.write(pred[i].strip() + "\n")


def summarize(raw_txt_fp, result_fp, model, max_length=3, max_pos=512, return_summary=True):
    model.eval()
    processed_text, full_length = preprocess(raw_txt_fp)
    input_data = load_text(processed_text, max_pos, device="cpu")
    test(model, input_data, result_fp, max_length, block_trigram=True)
    if return_summary:
        return open(result_fp).read().strip()
        