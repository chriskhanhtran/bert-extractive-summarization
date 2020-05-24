import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from transformers import BertModel, BertConfig
from models.encoder import ExtTransformerEncoder


class Bert(nn.Module):
    def __init__(self, finetune=True):
        super(Bert, self).__init__()
        configuration = BertConfig()
        self.model = BertModel(configuration)
        self.finetune = finetune

    def forward(self, x, segs, mask):
        if self.finetune:
            top_vec, _ = self.model(x, attention_mask=mask, token_type_ids=segs)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, attention_mask=mask, token_type_ids=segs)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, device, checkpoint, max_pos=512):
        super().__init__()
        self.device = device
        self.bert = Bert()
        self.ext_layer = ExtTransformerEncoder(
            self.bert.model.config.hidden_size, d_ff=2048, heads=8, dropout=0.2, num_inter_layers=2
        )

        if max_pos > 512:
            my_pos_embeddings = nn.Embedding(max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][
                None, :].repeat(max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint, strict=True)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
