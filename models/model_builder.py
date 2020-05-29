import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from transformers import BertModel, BertConfig, DistilBertConfig, DistilBertModel
from models.MobileBert.modeling_mobilebert import MobileBertConfig, MobileBertModel
from models.encoder import ExtTransformerEncoder

class Bert(nn.Module):
    def __init__(self, bert_type='bertbase'):
        super(Bert, self).__init__()
        self.bert_type = bert_type

        if bert_type == 'bertbase':
            configuration = BertConfig()
            self.model = BertModel(configuration)
        elif bert_type == 'distilbert':
            configuration = DistilBertConfig()
            self.model = DistilBertModel(configuration)           
        elif bert_type == 'mobilebert':
            configuration = MobileBertConfig.from_pretrained('checkpoints/mobilebert')
            self.model = MobileBertModel(configuration)  

    def forward(self, x, segs, mask):
        if self.bert_type == 'distilbert':
            top_vec = self.model(input_ids=x, attention_mask=mask)[0]
        else:
            top_vec, _ = self.model(x, attention_mask=mask, token_type_ids=segs)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, device, checkpoint=None, bert_type='bertbase'):
        super().__init__()
        self.device = device
        self.bert = Bert(bert_type=bert_type)
        self.ext_layer = ExtTransformerEncoder(
            self.bert.model.config.hidden_size, d_ff=2048, heads=8, dropout=0.2, num_inter_layers=2
        )

        if checkpoint is not None:
            self.load_state_dict(checkpoint, strict=True)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
