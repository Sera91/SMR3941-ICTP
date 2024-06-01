# -*- coding: utf-8 -*-
"""lecture_transformer_from_scratch_new.ipynb

This file generates locally the file .csv that you need after downloading the IMDB dataset


Disclaimers:

We will train in a standard way a Transformer Model. This is not BERT, which is a collection of Transformer layers. BERT is trained according to the Masked Language Model (MLM) paradigm.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install torchtext==0.6.0 torchdata
# 
# # After the installation, restart the session

import torch



#from torchtext.legacy import data, datasets

from torchtext import data, datasets
from torchtext import vocab
import numpy as np
import random, tqdm, sys, math, gzip
import os
import numpy as np
import pandas as pd
import random




batch_size = 4
sentence_length = 21 # Context Length, 512 for BERT






"""
 Data Preparation

One of the main concepts of TorchText is the `Field`. These define how your data should be processed. In our sentiment classification task the data consists of both the raw string of the review and the sentiment, either "pos" or "neg".

The parameters of a `Field` specify how the data should be processed.

We use the `TEXT` field to define how the review should be processed, and the `LABEL` field to process the sentiment.
"""



def get_IMDB_from_torchtext():
    SEED=42
    VOC_SIZE = 50000
    BATCH_SIZE = 4
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.LabelField(sequential=False, dtype = torch.float, batch_first=True)
    train, test = datasets.IMDB.splits(TEXT, LABEL)
    #TEXT.build_vocab(train, max_size=VOC_SIZE - 2)
    #LABEL.build_vocab(train)
    #train_iter, test_iter = data.Iterator.splits((train, test), batch_size=1, device = 'cpu')
    #train_iter, test_iter  = data.BucketIterator.splits((train, test), batch_size=BATCH_SIZE, device='cpu')
    #train_iter, valid_iter = train_iter.split(random_state = random.seed(SEED), split_ratio=0.8)

    labels, reviews = [], []
    for line in train.examples:
        #print(item.label, item.text )
        label=vars(line)['label']
        review=''.join(str(var)+" " for var in vars(line)['text'])
        assert label in ('pos', 'neg')
        #print(label)
        labels.append(label)
        reviews.append(review)
    df_train = pd.DataFrame({'sentiment': labels, 'review': reviews})
    df_train['sentiment'] = df_train['sentiment'].map({'pos': 1, 'neg': 0})
    print('original df_train.shape: ', df_train.shape)
    #df_train = df_train.drop_duplicates()
    #print('after drop_duplicates, df_train.shape: ', df_train.shape)
    labels, reviews = [], []
    for line in test.examples:
        label=vars(line)['label']
        review=vars(line)['text']
        assert label in ('pos', 'neg')
        labels.append(label)
        reviews.append(review)
    df_test = pd.DataFrame({'sentiment': labels, 'review': reviews})
    df_test['sentiment'] = df_test['sentiment'].map({'pos': 1, 'neg': 0})
    print('original df_test.shape: ', df_test.shape)
    #df_test = df_test.drop_duplicates()
    #print('after drop_duplicates, df_test.shape: ', df_test.shape)

    return df_train, df_test




df_train, df_test = get_IMDB_from_torchtext()

df_train.to_csv('./imdb_train.csv', index=True)
df_test.to_csv('./imdb_test.csv', index=True)



print(df_train.head(3))


