import pandas as pd
import numpy as np
import time
import datetime
from GoogleNews import GoogleNews
from dataclasses import dataclass
from transformers import TFBertModel
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from transformers import BertTokenizer

words_targets = ['TESLA', 'FORD', 'GENERAL MOTORS']

def build_model():
    model = TFBertModel.from_pretrained('bert-base-uncased')
    # two input layers, layers names must match dictionary keys in the dataset
    input_ids = tf.keras.layers.Input(shape=(64,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(64,), name='attention_mask', dtype='int32')

    # we access the transformer model within our bert object
    embeddings = model.bert(input_ids, attention_mask=mask)[1]

    # convert bert embeddings into 2 output classes with two dense layers
    x = tf.keras.layers.Dense(1024, activation='relu')(embeddings)
    y = tf.keras.layers.Dense(2, activation='sigmoid', name='outputs')(x)

    model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

    filepath = "model_best.hdf5"

    model.load_weights(filepath)
    return model

def gather_gnews():
    global words_targets

    res = {}

    # getting news from the last 24h
    start = datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(days=1),"%d/%m/%Y")
    end = datetime.datetime.strftime(datetime.datetime.now(),"%d/%m/%Y")
    # printing to know where we are in a long process of around 10 minutes
    googlenews = GoogleNews(start=start, end=end)
    for target in words_targets:
        googlenews.get_news(target)
        # adding the gathered articles to the ones already gathered from previous weeks
        results = googlenews.results()
        res[target] = pd.DataFrame(results)
        googlenews.clear()
        # and wait a little bit because google would ban too many calls in a small period of time
        time.sleep(1)

    return res

def one_inference(text, model):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.encode_plus(text, max_length=64, truncation=True, padding='max_length', add_special_tokens=True, return_token_type_ids=False, return_tensors='tf')
    to_send = {'input_ids': tf.cast(tokens['input_ids'], tf.float64),
               'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}
    vec = model.predict(to_send)[0]
    return max(vec)

def infer(news, model):
    alerts = []
    for target in words_targets:
        links = news['link']
        predictions = [one_inference(title, model) for title in news['title']]
        for i, prediction in enumerate(predictions):
            if prediction > 0.5:#arbitrary threshold
                alerts.append({'target':target, 'alert':prediction, 'title':links[i]})
    return pd.DataFrame(alerts)

if __name__ == '__main__':
    news = gather_gnews()
    model = build_model()
    alerts = infer(news, model)
    alerts.to_csv('daily_alerts.csv')#send the file elsewhere than on a docker container so that it can be read by other services
