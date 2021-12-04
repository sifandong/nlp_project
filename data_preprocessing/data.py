import json
import csv
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import unidecode
import contractions
from sklearn.preprocessing import LabelEncoder
from word2number import w2n
import spacy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer

def label(num):
    if num == 5.0 or num == 4.0:
        return 2
    elif num == 3.0:
        return 1
    else:
        return 0

# pre-processing function remove html tags
# def strip_html_tags(text):
#     """remove html tags from text"""
#     soup = BeautifulSoup(text, "html.parser")
#     stripped_text = soup.get_text(separator=" ")
#     return stripped_text
#
# # pre-processing function remove accented characters
# def remove_accented_chars(text):
#     """remove accented characters from text, e.g. café"""
#     text = unidecode.unidecode(text)
#     return text
#
# # pre-processing function expand shortened words
# def expand_contractions(text):
#     """expand shortened words, e.g. don't to do not"""
#     text = contractions.fix(text)
#     return text
#
# # pre-processing function remove stopwords
# def remove_stopwords(text):
#     en_sw = spacy.load('en_core_web_sm')
#     deselect_stop_words = ['no', 'not']
#     for w in deselect_stop_words:
#         en_sw.vocab[w].is_stop = False
nlp = spacy.load('en_core_web_sm')

# exclude words from spacy stopwords list
deselect_stop_words = ['no', 'not']
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False


def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)
    return text


def text_preprocessing(text, accented_chars=True, contractions=True,
                       convert_num=True, extra_whitespace=True,
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True,
                       stop_words=True):
    """preprocess text with default option set to true for all steps"""
    if remove_html == True:  # remove html tags
        text = strip_html_tags(text)
    if extra_whitespace == True:  # remove extra whitespaces
        text = remove_whitespace(text)
    if accented_chars == True:  # remove accented characters
        text = remove_accented_chars(text)
    if contractions == True:  # expand contractions
        text = expand_contractions(text)
    if lowercase == True:  # convert all characters to lowercase
        text = text.lower()

    doc = nlp(text)  # tokenise text

    clean_text = []

    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM':
            flag = False
        # remove punctuations
        if punctuations == True and token.pos_ == 'PUNCT' and flag == True:
            flag = False
        # remove special characters
        if special_chars == True and token.pos_ == 'SYM' and flag == True:
            flag = False
        # remove numbers
        if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
                and flag == True:
            flag = False
        # convert number words to numeric numbers
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # append tokens edited and not removed to list
        if edit != "" and flag == True:
            clean_text.append(edit)
    return clean_text



# extract columns
df_raw = pd.read_json('./Gift_Cards.json', lines=True)
# print(df_raw.info())
df_text = df_raw[['overall', 'reviewText']].copy()
# print(df_text.head())

df_text['label'] = df_text['overall'].apply(label)
# print(df_text.size())
df_text = df_text.dropna(axis=0, subset=['reviewText'])
# print(df_text.size())

# print(df_text.head(30))
# print(df_text.head())
# pre-processing
df_cleaned = df_text[['overall','reviewText','label']][:99].copy()
df_cleaned['cleaned_text'] = df_cleaned['reviewText'].apply(text_preprocessing)
# print(df_cleaned[:10]['cleaned_texts'])
# print(df_cleaned.head())
# print(df_cleaned.head())

# df_cleaned.to_csv('./cleaned.csv')
# df = pd.DataFrame(pd.read_csv('./cleaned.csv'))


msk = np.random.rand(len(df_cleaned)) < 0.8
df_train = df_cleaned[msk]
df_test = df_cleaned[~msk]
X_train = df_train["cleaned_text"]
Y_train = df_train["label"]
X_test = df_test["cleaned_text"]
Y_test = df_test["label"]

vectorizer = CountVectorizer()
X_real_train = []
for line in X_train:
    # print(type(line))
    # print(line)
    temp = vectorizer.fit_transform(line).toarray()
    X_real_train.append(temp)
for line in X_test:
    temp = vectorizer.fit_transform(line).toarray()
    X_real_train.append(temp)

print(X_real_train[:10])

# X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# print(type(X_train))
#
# clf = MultinomialNB()
# classifier_model = clf.fit(X_train,Y_train)
# print(np.mean(classifier_model.predict(X_test) == Y_test))
