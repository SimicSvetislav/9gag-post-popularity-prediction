# -*- coding: utf-8 -*-

from transformers import BertModel, BertTokenizer
import torch
from torch import nn

import numpy as np
import csv
import pandas as pd

INPUT_FILE = 'comment_data.csv'
OUTPUT_FILE = 'comments_data_sentiment.csv'
COMMENT_SCORES_FILE = 'comments_post_sentiment.csv'

BERT_MODEL_NAME = 'bert-base-cased'
PRETRAINED_MODEL = 'bert_pretrained.bin'
DROPOUT_RATE = 0.3

BATCH_SIZE = 1
MAX_LEN = 160

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

CLASS_NAMES = ['negative', 'neutral', 'positive']
CLASS_SCORES = [-1, 0, 1]

class SentimentPredictor(nn.Module):
    
    def __init__(self, n_classes):
        super(SentimentPredictor, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.drop = nn.Dropout(p=DROPOUT_RATE)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
loss_fn = nn.CrossEntropyLoss().to(device)

model = SentimentPredictor(len(CLASS_NAMES))

try:    
    model.load_state_dict(torch.load(PRETRAINED_MODEL))
except FileNotFoundError:
    raise Exception("Pretrained model not downloaded.\nPlease run 'gdown --id 1V8itWtowCYnb2Bc9KlK9SxGff9WwmogA' in Anaconda prompt.")

model = model.to(device)

def encode_comment(text):
    encoded_review = tokenizer.encode_plus(text, max_length=MAX_LEN,
      add_special_tokens=True, return_token_type_ids=False,
      pad_to_max_length=True, return_attention_mask=True,
      return_tensors='pt' # Tensors for pytorch
    )
    
    return encoded_review

def predict_pretrained_bert():
    
    print("Pytorch using", device)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as comments_file, open(OUTPUT_FILE, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        
        writer.writerow(['comment_id', 'post_id', 'sentiment', 'sentiment_score'])
        
        reader = csv.reader(comments_file)
        
        for i, row in enumerate(list(reader)[1:]):
            '''
            row[0] -> comment id
            row[1] -> post id
            row[3] -> text
            '''
            # print(row[3])
            encoded_text = encode_comment(row[3])
            
            input_ids = encoded_text['input_ids'].to(device)
            attention_mask = encoded_text['attention_mask'].to(device)
            
            output = model(input_ids, attention_mask)
            
            _, prediction = torch.max(output, dim=1)
            # print(f'Review text: {row[3]}')
            # print(f'Sentiment  : {CLASS_NAMES[prediction]}')
            
            writer.writerow([row[0], row[1], CLASS_NAMES[prediction], CLASS_SCORES[prediction]])
            
            if i%500==0:
                print(i)

def read_scores():
    
    with open("../../features_complete_v3.csv", "r") as features_file:
        dict_reader = csv.DictReader(features_file)
        
        scores_dict = {}
        for row in dict_reader:
            scores_dict[row["id"]] = row["log_score"]
            
        return scores_dict

def calcualte_post_comments_sentiment():
    
    with open(OUTPUT_FILE, 'r') as comment_scores_file:
        rows = list(csv.reader(comment_scores_file))[1:]

    post_counter = {}
    post_sums = {}

    for row in rows:
        '''
        row[0] -> comment id
        row[1] -> post id
        row[2] -> prediction
        row[3] -> prediction score
        '''
        
        post_id = row[1]
        
        if post_id in post_counter:
            post_counter[post_id] += 1
            post_sums[post_id] += int(row[3])
        else:
            post_counter[post_id] = 1
            post_sums[post_id] = int(row[3])
    
    scores_dict = read_scores()
    
    with open(COMMENT_SCORES_FILE, 'w', newline='') as scores_file:
        
        writer = csv.writer(scores_file)
        writer.writerow(['id', 'average_sentiment', 'score'])
        
        i = 0
    
        for post_id in post_counter:
            counter = post_counter[post_id]
            sentiment_sum = post_sums[post_id]
            
            if post_id not in scores_dict: 
                continue    
            
            score = scores_dict[post_id]
            
            avg_sentiment = sentiment_sum/counter
            
            writer.writerow([post_id, avg_sentiment, score])
            
            i+=1
            
            if i % 100 == 0:
                print(i)
        
def explore_scores():
    df = pd.read_csv('comments_data_sentiment.csv')
    print(df.describe())
    print(df['sentiment_score'].value_counts())

        
if __name__=='__main__':
    
    predict_pretrained_bert()
    
    calcualte_post_comments_sentiment()
    
    # explore_scores()
    
    pass
    