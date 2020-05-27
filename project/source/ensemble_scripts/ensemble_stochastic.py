# This file is created by SFU CMPT825 CPW Group
# The purpose of this file is implementation of ensemble stochastic 
from pyspark import SparkConf, SparkContext
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

threshold = 0.85

def chose_best_answer(x, y):
    x_model = x[0]
    y_model = y[0]

    # 80% chance to chose answer predicted from high performance model
    if (model_performance[x_model] > model_performance[y_model]) and random.random() < threshold:
        return x
    else:
        return y

def main(inputs, output):
    predictions_list = []
    os.chdir(inputs)
    for filename in glob.glob('*.json'):
        "assume predictions file format as bert_19000.json"
        mode_name = filename.split("_")[0]
        with open(filename, 'r') as f:
            pred = json.load(f)
            for key in pred:
                predictions_list.append((key,(mode_name,pred[key])))

    preds_rdd = sc.parallelize(predictions_list)
    best_ans_rdd = preds_rdd.reduceByKey(chose_best_answer)
    # best_ans = best_ans_rdd.sortBy(lambda x: int(x[0].split("\"")[0])).collect()
    best_ans = best_ans_rdd.sortByKey().collect()
    out_dict = {}
    for row in best_ans:
        out_dict[row[0]] = row[1][1]

    with open(output, 'w') as json_file:
        json.dump(out_dict, json_file)

if __name__ == '__main__':
    conf = SparkConf().setAppName('ensemble predictions')
    sc = SparkContext(conf=conf)
    sc.setLogLevel('WARN')
    assert sc.version >= '2.4'  # make sure we have Spark 2.4+
    inputs = sys.argv[1]
    output = "ensemble_stochastic_"+ str(threshold)+".json"
    main(inputs,output)