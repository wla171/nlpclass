# This file is created by SFU CMPT825 CPW Group, which reference the evaluate function in run_squad.py 
# The purpose of this file is implementation of prediction for DistilBERT Model
import os, torch, timeit
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm
from utils_squad import read_squad_examples, convert_examples_to_features, RawResult, write_predictions
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer

def do_prediction(model_dir):
    # 1. Load a trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model = DistilBertForQuestionAnswering.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # 2. Load and pre-process the test set

    dev_file = "data/sfu.json"
    predict_batch_size = 2
    max_seq_length = 384

    eval_examples = read_squad_examples(input_file=dev_file, is_training=False, version_2_with_negative=False)

    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=128,
                max_query_length=64,
                is_training=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=predict_batch_size)

    # 3. Run inference on the test set

    all_results = []
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader):

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():       
            batch_start_logits, batch_end_logits = model(input_ids, input_mask)
                
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                                 start_logits=start_logits,
                                                 end_logits=end_logits))
            
    output_prediction_file = os.path.join(model_dir, "predictions_sfu.json")
    output_nbest_file = os.path.join(model_dir, "nbest_predictions_sfu.json")
    output_null_log_odds_file = os.path.join(model_dir, "null_odds_sfu.json")

    preds = write_predictions(eval_examples, eval_features, all_results, 20,
                          30, True, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file, True,
                          False, 0.0)

def main():
    MODEL_DIR = './models/distilbert'
    for name in os.listdir(MODEL_DIR):
        if os.path.isdir(os.path.join(MODEL_DIR, name)):
            print('Do predictions on sfu.json for distilbert: ' + name)
            do_prediction(os.path.join(MODEL_DIR, name))

if __name__ == '__main__':
    start_time = timeit.default_timer()
    main()
    evalTime = timeit.default_timer() - start_time
    print("Evaluation done in total %f secs", evalTime)