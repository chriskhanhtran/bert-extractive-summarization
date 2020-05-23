import numpy as np
import torch
import transformers
from models.model_builder import ExtSummarizer
import time

def load_text(source_fp, max_pos, device):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)
    sep_vid = tokenizer.vocab['[SEP]']
    cls_vid = tokenizer.vocab['[CLS]']

    def _process_src(raw):
        raw = raw.strip().lower()
        raw = raw.replace('[cls]', '[CLS]').replace('[sep]', '[SEP]')

        # Tokenize and add special tokens to start and end of text
        src_subtokens = tokenizer.tokenize(raw)
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        # Convert to ids
        src_subtoken_idxs = tokenizer.convert_tokens_to_ids(src_subtokens)

        # Truncate
        src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
        src_subtoken_idxs[-1] = sep_vid

        # Add segment ids
        # Create a list of index of [SEP] tokens
        _segs = [-1] + \
            [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        segs = segs[:max_pos]
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        # Convert to Tensor
        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        # Create attention mask
        mask_src = (1 - (src == 0).float()).to(device)
        # Create a list of indexes of [CLS] tokens
        cls_ids = [[i for i, t in enumerate(
            src_subtoken_idxs) if t == cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0

        return src, mask_src, segments_ids, clss, mask_cls

    with open(source_fp) as source:
        x = source.read()
        src, mask_src, segments_ids, clss, mask_cls = _process_src(x)
        segs = torch.tensor(segments_ids)[None, :].to(device)
        src_text = [
            [sent.replace('[SEP]', '').strip() for sent in x.split('[CLS]')]]
        return src, mask_src, segs, clss, mask_cls, src_text

def test(model, input_data, result_path, block_trigram=True):
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(c, p):
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    with open(result_path, 'w') as save_pred:
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
                for j in selected_ids[i][:len(src_str[i])]:
                    if (j >= len(src_str[i])):
                        continue
                    candidate = src_str[i][j].strip()
                    if block_trigram:
                        if not _block_tri(candidate, _pred):
                            _pred.append(candidate)
                    else:
                        _pred.append(candidate)

                    if len(_pred) == 3:
                        break

                _pred = '<q>'.join(_pred)
                pred.append(_pred)

            for i in range(len(pred)):
                save_pred.write(pred[i].strip() + '\n')


def test_text_ext(raw_txt_fp, result_fp, checkpoint, device='cuda', max_pos=512):
    t0 = time.time()
    checkpoint = torch.load(checkpoint)
    model = ExtSummarizer(device, checkpoint, max_pos)
    model.eval()
    t1 = time.time()
    print(f'Loading checkpoint in {t1 - t0:.2f}')
    input_data = load_text(raw_txt_fp, max_pos, device)
    t2 = time.time()
    print(f'Processing data in {t2 - t1:.2f}')
    test(model, input_data, result_fp, block_trigram=True)
    t3 = time.time()
    print(f'Summarizing in {t3 - t2:.2f}')