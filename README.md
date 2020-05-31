# Extractive Summarization with BERT

In an effort to make BERTSUM ([Liu et al., 2019](https://github.com/nlpyang/PreSumm)) lighter and faster for low-resource devices, I fine-tuned DistilBERT ([Sanh et al., 2019](https://arxiv.org/abs/1910.01108)) and MobileBERT ([Sun et al., 2019](https://arxiv.org/abs/2004.02984)), two lite versions of BERT on CNN/DailyMail dataset. DistilBERT retains BERT-base's performance in extractive summarization while being 45% smaller. MobileBERT is 4x smaller and 2.7x faster than BERT-base yet retains 94% of its performance.

- **Demo with MobileBert:** [Open](https://extractive-summarization.herokuapp.com/)
- **Blog post:** [Read](https://chriskhanhtran.github.io/posts/extractive-summarization-with-bert/)

I modified the source codes in [PreSumm](https://github.com/nlpyang/PreSumm) to use the HuggingFace's `transformers` library and their pretrained DistilBERT model. At the moment (05/31/2020), MobileBERT is not yet available in `transformers`. Fortunately, I found this [PyTorch implementation of MobileBERT](https://github.com/lonePatient/MobileBert_PyTorch) by @lonePatient.

Please visit [PreSumm](https://github.com/nlpyang/PreSumm) for instructions for training on CNN/DailyMail and this [modified repo]() forked from PreSumm for fine-tuning DistilBERT and MobileBERT.

## Results on CNN/DailyMail

| Models     | ROUGE-1 |	ROUGE-2 | ROUGE-L | Inference Time* | Size   | Params   | Download |
|:-----------|:-------:|:--------:|:-------:|:---------------:|:------:|:--------:|:--------:|
| bert-base  | 43.23   | 20.24    | 39.63   | 1.65 s          | 475 MB | 120.5 M  | [link](https://www.googleapis.com/drive/v3/files/1t27zkFMUnuqRcsqf2fh8F1RwaqFoMw5e?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE) |
| distilbert | 42.84   | 20.04    | 39.31   | 925 ms          | 310 MB | 77.4 M   | [link](https://www.googleapis.com/drive/v3/files/1WxU7cHECfYaU32oTM0JByTRGS5f6SYEF?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE) |
| mobilebert | 40.59   | 17.98    | 36.99   | 609 ms          | 128 MB | 30.8 M   | [link](https://www.googleapis.com/drive/v3/files/1umMOXoueo38zID_AKFSIOGxG9XjS5hDC?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE) |

\**Average time running on a standard Google Colab notebook*

[**Tensorboard**](https://tensorboard.dev/experiment/Ly7CRURRSOuPBlZADaqBlQ/#scalars)
![](tensorboard.JPG)

## Setup
```sh
git clone https://github.com/chriskhanhtran/bert-extractive-summarization.git
cd bert-extractive-summarization
pip install -r requirements.txt
```

Download pretrained checkpoints:

```sh
wget -O "checkpoints/bertbase_ext.pt" "https://www.googleapis.com/drive/v3/files/1t27zkFMUnuqRcsqf2fh8F1RwaqFoMw5e?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"
wget -O "checkpoints/distilbert_ext.pt" "https://www.googleapis.com/drive/v3/files/1WxU7cHECfYaU32oTM0JByTRGS5f6SYEF?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"
wget -O "checkpoints/mobilebert_ext.pt" "https://www.googleapis.com/drive/v3/files/1umMOXoueo38zID_AKFSIOGxG9XjS5hDC?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"
```

## Usage
[![](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1hwpYC-AU6C_nwuM_N5ynOShXIRGv-U51#scrollTo=KizhzOxVOjaN)
```python
import torch
from models.model_builder import ExtSummarizer
from ext_sum import summarize

# Load model
model_type = 'mobilebert' #@param ['bertbase', 'distilbert', 'mobilebert']
checkpoint = torch.load(f'checkpoints/{model_type}_ext.pt', map_location='cpu')
model = ExtSummarizer(checkpoint=checkpoint, bert_type=model_type, device='cpu')

# Run summarization
input_fp = 'raw_data/input.txt'
result_fp = 'results/summary.txt'
summary = summarize(input_fp, result_fp, model, max_length=3)
print(summary)
```

## Samples

![](https://github.com/chriskhanhtran/minimal-portfolio/blob/master/images/bertsum.gif?raw=true)

**Original:** https://www.cnn.com/2020/05/22/business/hertz-bankruptcy/index.html

**bert-base**
```
The company has been renting cars since 1918, when it set up shop with a dozen Ford Model Ts, and has survived
the Great Depression, the virtual halt of US auto production during World War II and numerous oil price shocks.
By declaring bankruptcy, Hertz says it intends to stay in business while restructuring its debts and emerging a
financially healthier company. The filing is arguably the highest-profile bankruptcy of the Covid-19 crisis,
which has prompted bankruptcies by national retailers like JCPenney Neiman Marcus and J.Crew , along with some
energy companies such as Whiting Petroleum and Diamond Offshore Drilling .
```

**distilbert**
```
By declaring bankruptcy, Hertz says it intends to stay in business while restructuring its debts and emerging a
financially healthier company. But many companies that have filed for bankruptcy with the intention of staying
in business have not survived the process. The company has been renting cars since 1918, when it set up shop
with a dozen Ford Model Ts, and has survived the Great Depression, the virtual halt of US auto production during
World War II and numerous oil price shocks.
```

**mobilebert**
```
By declaring bankruptcy, Hertz says it intends to stay in business while restructuring its debts and emerging a
financially healthier company. The company has been renting cars since 1918, when it set up shop with a dozen
Ford Model Ts, and has survived the Great Depression, the virtual halt of US auto production during World War II
and numerous oil price shocks. "The impact of Covid-19 on travel demand was sudden and dramatic, causing an
abrupt decline in the company's revenue and future bookings," said the company's statement.
```

## References
- [1] [PreSumm:  Text Summarization with Pretrained Encoders](https://github.com/nlpyang/PreSumm)
- [2] [DistilBERT: Smaller, faster, cheaper, lighter version of BERT](https://huggingface.co/transformers/model_doc/distilbert.html)
- [3] [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://github.com/google-research/google-research/tree/master/mobilebert)
- [4] [MobileBert_PyTorch](https://github.com/lonePatient/MobileBert_PyTorch)
