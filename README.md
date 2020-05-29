**Demo:** https://bert-summarization.herokuapp.com/

**Results on CNN/DailyMail**

| Models     | ROUGE-1 |	ROUGE-2 | ROUGE-L | Avg. Inf. Time | Size   | Params   |
|:-----------|:-------:|:--------:|:-------:|:--------------:|:------:|:--------:|
| bert-base  | 43.23   | 20.24    | 39.63   | 1.06 s         | 475 MB | 120.5 M  |
| distilbert | 42.84   | 20.04    | 39.31   | 641 ms         | 310 MB | 77.4 M   |
| mobilebert | 40.59   | 17.98    | 36.99   | 462 ms         | 128 MB | 30.8 M   |

[**Tensorboard**](https://tensorboard.dev/experiment/wX89oBpMRyatmPwD0RQDOw/#scalars&_smoothingWeight=0.306)
