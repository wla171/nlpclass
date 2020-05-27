# This project baseline is used the open source library huggingface/transformers for pyTorch
https://github.com/huggingface/transformers/tree/master/examples

Training command 
```
python run_squad.py     --model_type albert     --model_name_or_path albert-base-v2     --do_train     --do_eval     --do_lower_case     --train_file train-v1.1.json     --predict_file dev-v1.1.json     --learning_rate 3e-5     --num_train_epochs 1 --save_steps 1000     --max_seq_length 384     --doc_stride 128     --output_dir models/kaggle/     --per_gpu_eval_batch_size=3       --per_gpu_train_batch_size=3 
```
You may want to download following data:
trian-v1.1.json
https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

dev-v1.1.json
https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

Google Natural Question Dataset
https://ai.google.com/research/NaturalQuestions