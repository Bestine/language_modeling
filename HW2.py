# your imports go here
import numpy as np
import csv
import spacy
from collections import Counter
import math
import nltk
from sentence_transformers import SentenceTransformer

# global variables (e.g., nlp modules, sentencetransformer models, dependencymatcher objects) go here
# adding globals here will mean that when we import e.g., extract_indirect_object it will still work ;)
# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('language_modeling_env/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-3.5.0')
model = SentenceTransformer('bert-base-nli-mean-tokens')

# TASK 1
def get_sentence_structure(sentence):
    sentence_structure = None
    
    doc = nlp(sentence)
    #Find the dependency matchers
    tokens_matchers = [matchers.dep_ for matchers in doc]
    
    if 'dative' in tokens_matchers and 'dobj' in tokens_matchers:
        # Now check if it is a PO
        if 'pobj'in tokens_matchers:
            #confirm again 
            if tokens_matchers.index("dobj") < tokens_matchers.index("dative"):
                sentence_structure = "PO"
        else:
            # Check for possible DO
            tokens_matchers.index("dative") < tokens_matchers.index("dobj")
            sentence_structure = "DO"
            
    else:
        sentence_structure =  None
    
    assert sentence_structure in {'DO', 'PO', None}
    return sentence_structure


def extract_direct_object(sentence):
    extracted_direct_object = ""
    # split the sentence into list
    tokens = sentence.split()
    
    doc = nlp(sentence)
    tokens_matchers = [matchers.dep_ for matchers in doc]
    
    # get the position of the direct object
    dobj_index = tokens_matchers.index("dobj")
    # now find the object
    direct_object_word = tokens[dobj_index]
    
    object_phrases = [chunk.text for chunk in doc.noun_chunks]
    # Get the full direct object 
    for chunk in object_phrases:
        if direct_object_word in chunk:
            extracted_direct_object = chunk
        else:
            direct_object = direct_object_word
    
    assert type(extracted_direct_object) is str
    return extracted_direct_object


def extract_indirect_object(sentence):
    extracted_indirect_object = None
    
    doc = nlp(sentence)
    tokens_matchers = [matchers.dep_ for matchers in doc]
    
    # get the sentence structure
    sentence_structure = get_sentence_structure(sentence)
        
    if sentence_structure == 'DO':
        indobj_index = tokens_matchers.index('dative')
        indirect_object_word = doc[indobj_index]
    elif sentence_structure == 'PO':
        indobj_index = tokens_matchers.index('pobj')
        indirect_object_word = doc[indobj_index]
    else:
        pass
    
    object_phrases = [chunk.text for chunk in doc.noun_chunks]
    #Get the indirect objects
    for chunk in object_phrases:
        if str(indirect_object_word) in chunk:
            extracted_indirect_object = chunk
            
    assert type(extract_indirect_object) is str
    return extracted_indirect_object


# TASK 2
def extract_feature_1(noun_phrase, sentence):
    # Using word length to extract feature from the noun phrase
    feature_1 = len(nltk.word_tokenize(noun_phrase))
    assert type(feature_1) is int
    return feature_1


def extract_feature_2(noun_phrase, sentence):
    position_tags = [position for (word, position) in nltk.pos_tag(nltk.word_tokenize(noun_phrase))]
    feature_2 = ''.join(pos_tags)
    assert type(feature_2) is str
    return feature_2


def extract_feature_3(noun_phrase, sentence):
    words = nltk.word_tokenize(noun_phrase)
    word_freqs = Counter(words)
    total_words = len(words)
    log_mean_freq = math.log(sum(word_freqs.values()) / total_words)
    log_median_freq = math.log(sorted(word_freqs.values())[total_words // 2])
    
    feature_3 = log_mean_freq
    assert type(feature_3) is float
    return feature_3

# TASK 3

def extract_sentence_embedding(sentence):
    sentence_embedding = model.encode(sentence)
    assert type(sentence_embedding) is np.array
    return sentence_embedding


def alter_sentence(sentence):
    altered_sentence = sentence
    # add anything to change the string here
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    
    # Get the part of speech of each word in the sentence
    pos_tags = nltk.pos_tag(tokens)
    
    # Choose a random word to replace
    word_to_replace = random.choice(pos_tags)[0]
    
    # Get the synset of the word
    synset = wordnet.synsets(word_to_replace)
    
    # Choose a random synset
    random_synset = random.choice(synset)
    
    # Get a random word from the chosen synset
    random_word = random_synset.lemmas()[random.randint(0, len(random_synset.lemmas())-1)].name()
    
    # Replace the chosen word with the random word
    altered_sentence = ' '.join([random_word if word == word_to_replace else word for word in tokens])

    return altered_sentence
