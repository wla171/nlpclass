## Predictions folder has testing result on both SQUAD and SFU dataset:

### 1. Testing result for ensemble models are in ens folder, please refer to our notebook for detailed model configuration. 

### 2. Testing result for individual models are in best folder.

### 3. nbest folder contains answer candidates used to produce ensemble results, scripts use to reproduce ensemble result are in source.zip

## To evaluate test result:

on sfu dataset, run command like
```
python3 evaluate-v1.1.py reference/sfu.json predictions/sfu/best/albert_22000_predictions_sfu.json
```
on squad dataset, run command like
```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
python3 evaluate-v1.1.py reference/dev-v1.1.json predictions/squad/best/albert_22000_predictions.json
```
## Reference folder has ground truth for SQUAD and SFU test set:

### you can also get SQUAD test set and save it to reference folder by run

```
cd reference
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

