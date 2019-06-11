#!/usr/bin/env python
# coding: utf8
"""Train NER for skills as a new entity
"""
import pandas as pd
import random
import spacy
from spacy.util import minibatch, compounding, decaying
from gensim.summarization import keywords
import string
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


# read in labeled courses
course = pd.read_csv('./data/courses_labeled.csv')

# combine title and description
course['text'] = course['title'] + ' ' + course['description']

# obtain start/end indices of skill tag in text
def get_start_end(text, skill):
    try:
        start = text.index(skill)
        end = start + len(skill)
    except:
        start = None
        end = None
    return start, end

course['idx'] = course.apply(lambda row: get_start_end(row['text'], row['skill']), axis=1)

# generate empty training labels
course['label'] = ''

# for courses that have exact match of skill tag in text, fill label with skill tag
course.loc[course['idx']!=(None, None), 'label'] = course['skill']

# clean text for courses that don't have exact match of skill tag in text
def remove_html(text):
    soup = BeautifulSoup(text)
    return soup.get_text()

course.loc[course['idx']==(None, None), 'text'] = course.loc[course['idx']==(None, None), 'text'].map(remove_html)

punctuations = string.punctuation
stopwords = stopwords.words('english')
nlp = spacy.load('en_core_web_sm')

def cleanup_text(doc):
    doc = nlp(doc, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    tokens = ' '.join(tokens)
    return tokens

course.loc[course['idx']==(None, None), 'text'] = course.loc[course['idx']==(None, None), 'text'].map(cleanup_text)


# generate keywords for non-exact matches
def get_keywords(text):
    return keywords(text,split=True)

course.loc[course['idx']==(None, None), 'label'] = course.loc[course['idx']==(None, None), 'text'].map(get_keywords)
course = course.loc[course.astype(str)['label']!='[]']

# choose the longest keyword string
course.loc[course['idx']==(None, None), 'label'] = course.loc[course['idx']==(None, None), 'label'].map(lambda x: max(x, key=len))

# get start/end indices for non-exact matches
course['idx'] = course.apply(lambda row: get_start_end(row['text'], row['label']), axis=1)
course = course[course['idx']!=(None,None)]

# save dataframe to csv
course.to_csv('./data/courses_textrank_labels.csv')

# generate dictionary for mapping labels to skills
label_to_skill = dict(zip(course['label'], course['skill']))

# generate training data
train = [(text, {'entities': [(id[0], id[1], 'SKILL')]}) for (text, id) in zip(course['text'], course['idx'])]

TRAIN_DATA = train


def train_NER(model=None, new_model_name="skill", output_dir='./models/train_textrank_labels', n_iter=5):
    # model = None for starting with an empty model
    # model = 'en_core_web_sm' for starting with a pretrained model
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label('SKILL')  # add new entity label to entity recognizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        dropout = decaying(0.2, 0, 0.02)
        loss_dict = {}
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.05, losses=losses) # or drop=next(dropout)
            loss_dict[itn] = losses['ner']
            print("Losses", losses)
        lists = sorted(loss_dict.items())
        x, y = zip(*lists)
        plt.plot(x, y)
        plt.show()

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

train_NER(model=None, new_model_name="skill", output_dir='./models/train_textrank_labels_1', n_iter=16)
