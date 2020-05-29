**Demo:** https://bert-summarization.herokuapp.com/

## Results on CNN/DailyMail

| Models     | ROUGE-1 |	ROUGE-2 | ROUGE-L | Avg. Inf. Time | Size   | Params   | Download |
|:-----------|:-------:|:--------:|:-------:|:--------------:|:------:|:--------:|:--------:|
| bert-base  | 43.23   | 20.24    | 39.63   | 1.06 s         | 475 MB | 120.5 M  |-|
| distilbert | 42.84   | 20.04    | 39.31   | 641 ms         | 310 MB | 77.4 M   |-|
| mobilebert | 40.59   | 17.98    | 36.99   | 424 ms         | 128 MB | 30.8 M   |-|

[**Tensorboard**](https://tensorboard.dev/experiment/wX89oBpMRyatmPwD0RQDOw/#scalars&_smoothingWeight=0.306)

## Usage
**Setup**
```sh
git clone https://github.com/chriskhanhtran/bert-summarization.git
cd bert-summarization
pip install -r requirements.txt
```

If you want to use MobileBert, do following:
```sh
cd ..
git glone https://github.com/lonePatient/MobileBert_PyTorch.git
```

**Summaziation**
```python
import torch
from models.model_builder import ExtSummarizer
from ext_sum import summarize

# Load model
model_type = 'bertbase' #@param ['bertbase', 'distilbert', 'mobilebert']
checkpoint = torch.load(f'checkpoint/{model_type}_ext.pt')
model = ExtSummarizer(device="cpu", checkpoint=checkpoint, bert_type=model_type, max_pos=512)

# Run summarization
input_fp = 'raw_data/news_from_url.txt'
result_fp = 'results/summary.txt'
summary = summarize(input_fp, result_fp, model, max_length=3)
print(summary)
```

## Samples
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
