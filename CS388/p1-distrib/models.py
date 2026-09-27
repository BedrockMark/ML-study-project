# models.py
import re
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence.
    The steps I'd like to take:
    1. lowercase all words
    2. keep only letters
    3. throw out low count words
    4. throw out short words
    """

    def __init__(self, indexer: Indexer):
        # --- Config ---
        # self.DROP_OUT_PERC = 0.1
        self.DROP_OUT_COUNT = 3
        
        # --- Forward declearation ---
        self.indexer:Indexer = indexer
        
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        count: Counter = Counter()
        words = [
            clean_word
            for word in sentence
            if len(word)>=3 and len(clean_word := re.sub(r"[^a-z\d]", "", word))
        ] # words after process
        if add_to_indexer:
            for word in words:
                count[word] += 1
            # print(f"[BOW] Count was: {len(count)}, now is {round(len(count)*(1-self.DROP_OUT_PERC))}")
            # count = count.most_common(round(len(count)*(1-self.DROP_OUT_PERC)))
            self.indexer = Indexer(k if count[k]>self.DROP_OUT_COUNT else None for k in count)
            count = self.extract_features(sentence)
        else:
            for word in words:
                count[self.indexer.index_of(word)] += 1
            count[-1] = 0
        return count


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class LogisticRegressionClassifier(SentimentClassifier, nn.Module):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    
    # 2. when turning into vectors: accumulate first, then softmax
    def __init__(self, feat_extractor: FeatureExtractor, vocab_size:int):
        super(LogisticRegressionClassifier, self).__init__()
        # External resource
        self.extractor:FeatureExtractor = feat_extractor
        self.vocab_size = vocab_size
        # runtime
        self.linear = nn.Linear(vocab_size, 2)
    
    def words_to_vec(self, words:List[str]) -> torch.Tensor:
        vec = torch.zeros(self.vocab_size, dtype=torch.float32)
        
        count = self.extractor.extract_features(words)
        if not count: return vec
        
        indices = torch.tensor(list(count.keys()), dtype=torch.long)
        values = torch.tensor(list(count.values()), dtype=torch.float32)
        vec[indices] = values
        
        return vec
    
    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        self.eval()
        with torch.no_grad():
            x = self.words_to_vec(ex_words)
            logits = self(x)

        return torch.argmax(logits, dim=-1)
        
    def forward(self, bow_vec):
        return self.linear(bow_vec)

def collate_batch(batch, model):
    # batch is a list with batch_size count of item
    bow_list = []
    labels = []
    for item in batch:
        vec = model.words_to_vec(item.words)
        bow_list.append(vec)
        labels.append(item.label)

    bow_tensor = torch.stack(bow_list)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    return bow_tensor, label_tensor

def create_collate_fn(model):
    return lambda batch: collate_batch(batch=batch, model=model)

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # Summon bag-of-word
    bow: List[str] = []
    for item in train_exs:
        bow += item.words
    _ = feat_extractor.extract_features(bow,True)
    vocab_size = len(_)
    print(f"After training, we have kept {vocab_size} words!")
    
    # Start training
    model = LogisticRegressionClassifier(feat_extractor=feat_extractor, vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.15)
    
    train_loader = DataLoader(
        train_exs,
        batch_size=32,
        shuffle=True,
        collate_fn=create_collate_fn(model=model),
    )
    
    for epoch in range(30):
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        for bow_vec, targets in train_loader:
            optimizer.zero_grad()
            output = model(bow_vec)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
            
        avg_loss = total_loss / total_samples
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            
    return model


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        raise NotImplementedError


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    raise NotImplementedError
