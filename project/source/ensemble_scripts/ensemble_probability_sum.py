# This file is created by SFU CMPT825 CPW Group
# The purpose of this file is implementation of ensemble probability sum 
import sys, json, glob, random, os
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

"{ model_name : avg F1 score on SQUAD}"
model_performance = {
    "bert":(88.26+88.26+88.08)/3,
    "albert":(88.85+89.38+89.40)/3,
    "distilbert":(85.03+85.50+85.21)/3,
    "albertAug":89.46,
    "albertGoogle":89.46
    }

def main(inputs, output): 
    all_nbest_preds = {}
    best_preds = {}

    # 1. load all nbest prediction file from every models
    os.chdir(inputs)
    for filename in glob.glob('*.json'):
        "assume predictions file format as bert_19000.json"
        mode_name = filename.split("_")[0]
        with open(filename, 'r') as f:
            nbest_pred = json.load(f)
            for question_id in nbest_pred:
                for ans in nbest_pred[question_id]:
                    ans["model"] = mode_name

                if question_id in all_nbest_preds:
                    # entend existing answer
                    all_nbest_preds[question_id].extend(nbest_pred[question_id])
                else:
                    all_nbest_preds[question_id] = nbest_pred[question_id]
    # 2. load original questions list

    # 3. iterate over all questions
    for qid in all_nbest_preds:
        ans_list = all_nbest_preds[qid]
        # count votes
        counts = {}
        for ans in ans_list:
            counts[ans["text"]] = counts.get(ans["text"], 0) + ans["probability"] * model_performance[ans["model"]]

        # select most voted answer
        best_ans = max(counts.items(), key=lambda x: x[1])[0]
        best_preds[qid] = best_ans

    # output best answer
    with open(output, 'w') as json_file:
        json.dump(best_preds, json_file)

if __name__ == '__main__':
    inputs = sys.argv[1]
    output = "ensemble_probability_sum.json"
    main(inputs,output)