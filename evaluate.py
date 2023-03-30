from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import numpy as np
from collections import Counter
import HW2

TRAINING_CORPUS = 'training_sentences.txt'

def main():
    train_sentences = load_corpus(TRAINING_CORPUS)
    # training
    train_Xs, _, _ = extract_features_all_sentences(train_sentences)
    train_ys = get_classifications(train_sentences)
    trained_model = train_model(train_Xs, train_ys)
    predictions = trained_model.predict(train_Xs)
    training_f1 = f1_score(predictions, train_ys)
    print(f"Your F1 score for predicting sentence order on the training data is {training_f1}")


def load_corpus(filename):
    sentences = []
    with open(filename, 'r') as f:
        for l in f:
            sentences.append(l.rstrip())
    return sentences


def extract_numeric_features(NP):
    int_feature = HW2.extract_feature_1(NP)
    float_feature = HW2.extract_feature_3(NP)
    return int_feature, float_feature


def extract_string_features(NP):
    string_features = HW2.extract_feature_2(NP)
    return string_features


def concatenate_features(
        DO_numeric_features,
        DO_string_features,
        PO_numeric_features,
        PO_string_features
        ):
    # should just work with a simple hstack, or concatenate axis=0
    DO_features = np.concat([DO_string_features, DO_numeric_features])
    PO_features = np.concat([PO_string_features, PO_numeric_features])
    all_features = np.concat([DO_features, PO_features])
    return all_features


def extract_features_all_sentences(
        list_of_sentences,
        mode='train',
        DO_cv=None,
        PO_cv=None
        ):
    DO_numeric_features, PO_numeric_features = [], []
    DO_string_features, PO_string_features = [], []
    for s in list_of_sentences:
        DO_NP = HW2.extract_direct_object(s)
        PO_NP = HW2.extract_indirect_object(s)
        DO_numeric_features.append(
            extract_numeric_features(DO_NP)
        )
        PO_numeric_features.append(
            extract_numeric_features(PO_NP)
        )
        DO_string_features.append(
            extract_string_features(DO_NP)
            )
        PO_string_features.append(
            extract_string_features(PO_NP)
        )
    if mode=='train':
        DO_cv, DO_string_features = vectorize_train(DO_string_features)
        PO_cv, PO_string_features = vectorize_train(PO_string_features)
    else:
        DO_string_features = vectorize_test(DO_string_features, DO_cv)
        PO_string_features = vectorize_test(PO_string_features, PO_cv)
    all_features = concatenate_features(
        DO_numeric_features,
        DO_string_features,
        PO_numeric_features,
        PO_string_features
        )
    if mode=='train':
        return all_features, DO_cv, PO_cv
    else:
        return all_features


def vectorize_train(list_of_strings: list):
    vectorizer = CountVectorizer(min_df=5)
    counts = Counter(list_of_strings)
    strings_w_unks = []
    for w in counts:
        if counts[w] < 5:
            strings_w_unks.append('UNK')
        else:
            strings_w_unks.append(w)
    vectorized = vectorizer.fit_transform(
        strings_w_unks
        ).todense().squeeze()
    return vectorizer, vectorized


def vectorize_test(list_of_strings, vectorizer):
    strings_w_unks = []
    for w in list_of_strings:
        if w not in vectorizer.vocab:
            strings_w_unks.append("UNK")
        else:
            strings_w_unks.append(w)
    vectorized = vectorizer.transform(
        strings_w_unks
        ).todense().squeeze()
    return vectorized


def get_classification(sentence):
    y = HW2.get_sentence_structure(sentence)
    return y


def get_classifications(list_of_sentences):
    ys = []
    for s in list_of_sentences:
        ys.append(get_classification(s))
    return ys


def train_model(X_train, y_train):
    clf = MLPClassifier(
        alpha=.05, random_state=1, max_iter=300,
        hidden_layer_sizes=(10, 10,),
        batch_size=1
        ).fit(X_train, y_train)
    return clf