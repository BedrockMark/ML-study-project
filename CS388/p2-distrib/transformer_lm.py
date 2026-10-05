# models.py

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from transformer import PositionalEncoding

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)


class NeuralLanguageModel(LanguageModel, nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, num_layers):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model, num_positions, True)
        
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                batch_first=True,
                dim_feedforward=256
            ),
            num_layers=num_layers
        )

        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, indecies):
        # Input -> [batch_size, seq_len(str of 20)]
        causal_mask = torch.triu(torch.full((indecies.size(1), indecies.size(1)), float('-inf')), diagonal=1)
        x = self.pos_embed(self.char_embed(indecies))
        logits = self.tf(x,mask=causal_mask)
        return nn.functional.log_softmax(self.output_proj(logits),dim=-1)
        
    def get_next_char_log_probs(self, context):
        self.eval()
        # print("Evaluating: ", context)
        with torch.no_grad():
            res = self(torch.tensor(str_to_indecies(' '+context),dtype=torch.long).reshape(1, -1))[-1][-1].numpy()
            # print("Result: ", res, res.shape)
            return res

def str_to_indecies(input:str):
    ret = []
    for i in input:
        if i == ' ': ret.append(26)
        else: ret.append(ord(i)-ord('a'))
    return ret

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    train_indecies = str_to_indecies(train_text)
    
    # Chunking
    CHUNK_LENGTH = 32
    input = []
    for i in range(len(train_indecies)-CHUNK_LENGTH):
        input.append((torch.tensor([26]+train_indecies[i:i+CHUNK_LENGTH-1]),torch.tensor(train_indecies[i:i+CHUNK_LENGTH])))
    
    # Batching
    data = DataLoader(
        shuffle=True,
        dataset=input,
        batch_size=256,
    )
    
    # Training
    # print(len(vocab_index))
    model = NeuralLanguageModel(len(vocab_index),CHUNK_LENGTH,64,2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = nn.NLLLoss()
    
    model.train()
    
    num_epochs = 5
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        
        for d in data:
            model.zero_grad()
            prob = model(d[0])
            # print(prob.shape, d[1].shape)
            loss = loss_fcn(prob.reshape(-1, prob.size(-1)), d[1].reshape(-1))
            loss.backward()
            optimizer.step()
            
            loss_this_epoch += loss.item()
        
        print(f"Loss in Epoch #{t} = {loss_this_epoch}")
    
    model.eval()
    return model
