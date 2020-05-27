## ensemble_scripts folder has three different ensemble algorithms we wrote:

to reproduce our ensemble result, run commands like

```
python3 ensemble_probability_sum.py [directory that only contains *_nbest_prodiction.json]
```

## predict_scripts folder

It has code we wrote to use bert, albert and distilbert model with weights obtained throught training, to do prediction on squad/sfu dataset. Due to submission size limit, we did not include our learned model weight. However, our result can be reproduced using our training parameter settings mentioned in report.

to do prediction, run command like 

```
python3 predict_albert.py [directory of your model weights] [directory of your test data]
```