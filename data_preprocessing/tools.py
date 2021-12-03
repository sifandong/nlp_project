import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import nltk
import re
import string
nltk.download("stopwords")



with open('Gift_Cards.json', encoding='utf-8-sig') as f_input:
    df = pd.read_json(f_input, lines=True)

df.to_csv('uncleaned.csv', encoding='utf-8', index=False)

def classify(x):
    if x == 5.0 or x==4.0:
        return 2
    if x==3.0:
        return 1
    return 0


def clean_dataframe(df):
    # creates new column with corresponding class labels, the output variable.
    df['y'] = df['overall'].apply(classify)

    # dropping uneccesary columns for the analysis
    df = df.drop(
        labels=['Unnamed: 0', 'verified', 'asin', 'style', 'reviewerName', 'description', 'title', 'rank', 'main_cat'],
        axis=1)

    # dropping all NaN values from the column reviewText
    df = df.dropna(axis=0, subset=['reviewText'])
    return df

def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct

def text_process(df):
  # Removing all the punctuations from the words, and changing the words to lower case to maintain uniformity
  df['reviewText']=df['reviewText'].apply(lambda x: remove_punctuation(x.lower()))
  # stemming
  stemmer = PorterStemmer()
  # stop words are the words like "the, I, our etc"
  words = stopwords.words("english")
  df['cleaned_reviews'] = df['reviewText'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
  return df

def process_df(df):
  df = clean_dataframe(df)
  df = text_process(df)
  return df

URL_UNCLEAN = "./uncleaned.csv"
URL_CLEAN = "./cleaned.csv"

def main(URL_CLEAN=URL_CLEAN, URL_UNCLEAN=URL_UNCLEAN):
  df_unclean = pd.read_csv(URL_UNCLEAN)
  print("UNCLEANED DATASET HEAD:\n",df_unclean.head(),"\n")

  df_clean = process_df(df_unclean)
  print("CLEANED DATASET HEAD:\n",df_clean.head(),"\n")

  #writing to dataframe
  df_clean.to_csv(URL_CLEAN)

main()