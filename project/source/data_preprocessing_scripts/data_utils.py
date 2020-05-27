# This file is created by SFU CMPT825 CPW Group
# The purpose of this file is filter out the long answer from Google Natural Question Dataset
# Also, include a helpful function which used in data augumentation

import os
import json
import nltk
import joblib
import re
from nltk.corpus import wordnet 
# !!! important, you may need to comment out the below code for downloading the wordnet model
# nltk.download("wordnet")

def replace_synonyms_words(word):
    # Then, we're going to use the term "program" to find synsets like so: 
    synonyms_word = wordnet.synsets(word) 
    if(len(synonyms_word) < 2):
        return word
    else:
        new_word = synonyms_word[1].lemmas()[0].name()
        return new_word

# Count number of samples which should match the number of short question span as Google mentioned
# "Short answers can be sets of spans in the document (106,926)"
# https://github.com/google-research-datasets/natural-questions
def countSamples(input_file):
    count = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        for line in reader: 
            count = count + 1
    return count

# function for getting short anwser only data from Google Natural Question dataset
def filter_long_answer(train_input_file, output_file):
    with open(output_file, 'w') as new_json:
        with open(train_input_file, "r", encoding='utf-8') as reader:
            for line in reader:
                input_data = json.loads(line)
                short_answer = input_data['annotations'][0]['short_answers']
                # Skipping data with no short answer
                if(len(short_answer) < 1):
                    continue
                new_json.write(line)
                if(write_count == 1):
                    break

# This function is created for reading Google NQ dataset, which should be used in utils_squad
def read_nq_examples(input_file, is_training, version_2_with_negative):
    """Read a Google Short Answer json file into a list of SquadExample."""
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    examples = []
    is_impossible = False
    use_cached = False
    if(use_cached): 
        examples = joblib.load('cached_example.sav')
    else: 
        with open(input_file, "r", encoding='utf-8') as reader:
            for line in reader:
                input_data = json.loads(line)
                paragraph_text = input_data["document_text"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)
                start_position = None
                end_position = None
                orig_answer_text = None
                qas_id = input_data["example_id"]
                question_text = input_data["question_text"]
                if is_training:
                    long_answer = input_data["annotations"][0]["long_answer"]
                    short_answers = input_data["annotations"][0]["short_answers"]
                    if (len(short_answers) < 1 and not is_impossible):
                        raise ValueError(
                                    "For training, each question should have exactly 1 answer.")
                    short_answer = short_answers[0]
                    context_text = doc_tokens[long_answer["start_token"]:long_answer["end_token"]]

                    start_position = short_answer["start_token"] - long_answer["start_token"]
                    end_position = short_answer["end_token"] - long_answer["start_token"]

                    non_html_tokens = []
                    html_index = []
                    count = 0
                    for token in context_text:
                        html_index.append(count)
                        if (re.match(r'<[^>]+>', token)):
                            count = count + 1
                            continue
                        else:
                            non_html_tokens.append(token)
                    
                    no_html_paragraph_text= ' '.join(non_html_tokens)
                    answer = short_answers[0]
                    orig_answer_text = " ".join(doc_tokens[answer["start_token"]:answer["end_token"]])
                    
                    start_position = start_position - html_index[start_position]
                    end_position = end_position -  html_index[end_position]

                    new_answer = " ".join(non_html_tokens[start_position:end_position])
                    example = SquadExample(
                            qas_id=qas_id,
                            question_text=question_text,
                            doc_tokens=non_html_tokens,
                            orig_answer_text=new_answer,
                            start_position=start_position,
                            end_position=end_position,
                            is_impossible=is_impossible)
                    examples.append(example)

        logger.info("Number of examples %s", len(examples))
        joblib.dump((examples), 'cached_example.sav')  
    return examples