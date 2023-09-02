import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import time
from rnn_layers_torch import *
from load_data import *
from pprint import pprint

class RNN(nn.Module):

    def __init__(self, word_to_idx, wordvec_dim, hidden_dim, cell_type, seed, device):
        super(RNN, self).__init__()

        vocab_size = len(word_to_idx)
        self.start_token = word_to_idx["<START>"]
        self.null_token = word_to_idx["<NULL>"]
        self.end_token = word_to_idx["<END>"]
        self.cell_type = cell_type
        self.params = {}

        if(seed is not None):
            np.random.seed(seed)

        self.params["W_embed"] = np.random.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = np.zeros(dim_mul * hidden_dim)

        self.params["W_vocab"] = np.random.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = np.zeros(vocab_size)

        for key in self.params.keys():
            self.params[key] = self.params[key].astype(np.float32)
            self.params[key] = torch.from_numpy(self.params[key])
            self.params[key] = self.params[key].to(device)
            self.params[key].requires_grad = True


    def forward(self, captions, h_init):
        '''returns all the hidden states of the RNN as a tensor of shape (N, T, H)
        '''

        captions_in = captions[:,:-1]
        N = captions.shape[0]
        # h0 = torch.tile(h_init, (N, 1))
        h0 = h_init
        h = None

        # Generate word embeddings from captions
        inputs = word_embedding_forward(captions_in, self.params["W_embed"])

        # RNN forward pass
        if(self.cell_type == "rnn"):
            h = rnn_forward(inputs, h0, self.params["Wx"], self.params["Wh"], self.params["b"])
        elif(self.cell_type == "lstm"):
            h = lstm_forward(inputs, h0, self.params["Wx"], self.params["Wh"], self.params["b"])
        else:
            return None

        return h
    
    def load(self, parameters):
        for key in self.params.keys():
            self.params[key] = parameters[key]
            self.params[key].requires_grad = True

class RNNAttention(nn.Module):

    def __init__(self, word_to_idx, wordvec_dim, hidden_dim, cell_type, seed, device):
        super(RNNAttention, self).__init__()

        vocab_size = len(word_to_idx)
        self.start_token = word_to_idx["<START>"]
        self.null_token = word_to_idx["<NULL>"]
        self.end_token = word_to_idx["<END>"]
        self.cell_type = cell_type
        self.params = {}

        if(seed is not None):
            np.random.seed(seed)

        self.params["W_embed"] = np.random.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = np.random.randn(wordvec_dim, dim_mul * hidden_dim) * 0.01
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = np.random.randn(hidden_dim, dim_mul * hidden_dim) * 0.01
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = np.zeros(dim_mul * hidden_dim)

        self.params["W_vocab"] = np.random.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = np.zeros(vocab_size)

        self.params["W_attention1"] = np.random.randn(hidden_dim * 2, hidden_dim) * 0.001
        self.params["W_attention1"] /= np.sqrt(hidden_dim * 2)
        self.params["b_attention1"] = np.zeros(hidden_dim)
        self.params["W_attention2"] = np.zeros((hidden_dim, 1))
        self.params["W_context"] = np.random.randn(hidden_dim, dim_mul * hidden_dim)

        for key in self.params.keys():
            self.params[key] = self.params[key].astype(np.float32)
            self.params[key] = torch.from_numpy(self.params[key])
            self.params[key] = self.params[key].to(device)
            self.params[key].requires_grad = True


    def forward(self, hidden_enc_states, captions, h_init, mask):
        '''returns all the hidden states of the RNN as a tensor of shape (N, T, H)
        '''

        captions_in = captions[:,:-1]
        N = captions.shape[0]
        # h0 = torch.tile(h_init, (N, 1))
        h0 = h_init
        h = None

        # Generate word embeddings from captions
        inputs = word_embedding_forward(captions_in, self.params["W_embed"])

        # RNN forward pass
        if(self.cell_type == "rnn"):
            h = rnn_attention_forward(inputs, h0, self.params["Wx"], self.params["Wh"], self.params["b"], \
                                      hidden_enc_states, self.params["W_attention1"],\
                                         self.params["b_attention1"], self.params["W_attention2"], mask, self.params["W_context"])
        elif(self.cell_type == "lstm"):
            h = lstm_attention_forward(inputs, h0, self.params["Wx"], self.params["Wh"], self.params["b"], \
                                      hidden_enc_states, self.params["W_attention1"],\
                                         self.params["b_attention1"], self.params["W_attention2"], mask, self.params["W_context"])
        else:
            return None

        return h
    
    def load(self, parameters):
        for key in self.params.keys():
            self.params[key] = parameters[key]
    
class Encoder(nn.Module):
    def __init__(self, word_to_idx, wordvec_dim, hidden_dim, cell_type, seed, device):
        super(Encoder, self).__init__()

        self.RNN = RNN(word_to_idx, wordvec_dim, hidden_dim, cell_type, seed, device)
        self.h_init = torch.randn(hidden_dim, requires_grad=True)
    
    def forward(self, captions):
        ''' captions are of shape (N, T + 1)
        '''
        h0 = torch.tile(self.h_init, (captions.shape[0], 1))
        hidden_states = self.RNN(captions, h0) # (N, T, H)
        mask = captions[:,1:] != self.RNN.null_token
        # Only returns the end hidden state (the state till the <END> token)
        end = torch.where(captions[:,1:] == self.RNN.end_token)
        return hidden_states, hidden_states[end], mask

class Decoder(nn.Module):
    def __init__(self, word_to_idx, wordvec_dim, hidden_dim, cell_type, seed, device):
        super(Decoder, self).__init__()

        self.RNN = RNNAttention(word_to_idx, wordvec_dim, hidden_dim, cell_type, seed, device)
        # Attention layers for calculating the attention between hidden state of decoder
        # and all the hidden states of the encoder.
        # The 2nd dimension of attention_w can be arbitrary
        
    def forward(self, captions, h_init, hidden_enc_states, mask):
        ''' captions are of shape (N, T + 1)
        '''
        hidden_states = self.RNN(hidden_enc_states, captions, h_init, mask) # (N, T, H)
        # assume the sentence starts with a <START> token and ends with a <END> token
        captions_out = captions[:,1:]

        # don't consider the loss where the token is <NULL>
        mask = captions_out != self.RNN.null_token

        out = temporal_affine_forward(hidden_states, self.RNN.params["W_vocab"], self.RNN.params["b_vocab"])
        loss = temporal_softmax_loss(out, captions_out, mask)
        return loss

class NMT(nn.Module):
    def __init__(self, word_to_idx_enc, word_to_idx_dec, wordvec_dim, hidden_dim, cell_type, seed, device):
        super(NMT, self).__init__()

        # TODO -> allow for different hidden states dimensions for encoder and decoder

        self.encoder = Encoder(word_to_idx_enc, wordvec_dim, hidden_dim, cell_type, seed, device)
        self.decoder = Decoder(word_to_idx_dec, wordvec_dim, hidden_dim, cell_type, seed, device)
        self.params = {}
        self.params["W_mid"] = torch.randn(hidden_dim, hidden_dim, requires_grad=True)
        self.params["b_mid"] = torch.randn(hidden_dim, requires_grad=True)

    def forward(self, captions1, captions2):
        ''' Both captions1 and captions2 start with <START> and end with <END> and are of shape (N, T + 1) (T is different for both)
        '''
        hidden_states, h_encoder, mask = self.encoder(captions1) # (N, T, H) and (N, H) and (N, H)
        # h_init = self.mid_layer(h_encoder)
        h_init = h_encoder @ self.params["W_mid"] + self.params["b_mid"]
        loss = self.decoder(captions2, h_init, hidden_states, mask)
        return loss
    
    def load(self, weights):
        self.encoder.RNN.load(weights["params_enc"])
        self.decoder.RNN.load(weights["params_dec"])
        for key in self.params.keys():
            self.params[key] = weights["mid_layer"][key]



file_en = "english.txt"
file_es = "spanish.txt"
lwflag = 0 # 0 for words, 1 for letter
word_to_idx = None

word_to_idx_en = make_dict(file_en)
word_to_idx_es = make_dict(file_es)

reverse_dict_en = {}
reverse_dict_es = {}

for keys, value in word_to_idx_en.items():
    reverse_dict_en[value] = keys

for keys, value in word_to_idx_es.items():
    reverse_dict_es[value] = keys

# All the tensors will be allocated by default on the device
# Except some cases where we are using torch.from_numpy. Why?

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Common Parameters For Both Encoder and Decoder
# TODO make the parameters different for both of them, define separately
hidden_dim = 256
word_vec_dim = 256
seed = 4
architecture = "lstm"
epochs = 100000
learning_rate = 0.001
# Create a NTM instance
nmt = NMT(word_to_idx_en, word_to_idx_es, word_vec_dim, hidden_dim, architecture, seed, device)

# Uncomment below lines and comment line in the for loop for cheking overfitting
# data_en, data_es = load_data_nmt(word_to_idx_en, word_to_idx_es, file_en, file_es, lines_count=1, max_train=30, hardcode=1)

# for i in range(len(data_en)):
#     words = [reverse_dict_en[val] for val in data_en[i]]
#     for word in words:
#         print(word + " ", end = "")

# print("")

# for i in range(len(data_es)):
#     words = [reverse_dict_es[val] for val in data_es[i]]
#     for word in words:
#         print(word + " ", end = "")

# print("")

# data_en = torch.from_numpy(data_en)
# data_es = torch.from_numpy(data_es)
# data_en = data_en.to(device)
# data_es = data_es.to(device)

weights_file = "check.pt"
# loaded_weights = torch.load(weights_file, map_location=device)
# nmt.load(loaded_weights)

parameters = []

for key in nmt.encoder.RNN.params.keys():
    parameters.append(nmt.encoder.RNN.params[key])

for key in nmt.decoder.RNN.params.keys():
    parameters.append(nmt.decoder.RNN.params[key])

for key in nmt.params.keys():
    parameters.append(nmt.params[key])

optimizer = optim.Adam(parameters, lr=learning_rate)
# optimizer.load_state_dict(loaded_weights["optime"])

for i in range(epochs):
    
    data_en, data_es = load_data_nmt(word_to_idx_en, word_to_idx_es, file_en, file_es, lines_count=1, max_train=20)
    data_en = torch.from_numpy(data_en)
    data_es = torch.from_numpy(data_es)
    data_en = data_en.to(device)
    data_es = data_es.to(device)

    loss = nmt(data_en, data_es)

    print(loss)

    for param in parameters:
        param.retain_grad()

    optimizer.zero_grad() 
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    # for param in parameters:
        # print(param.grad)
    optimizer.step()

    if i%100 == 0:
        torch.save({
            "params_enc" : nmt.encoder.RNN.params,
            "params_dec" : nmt.decoder.RNN.params,
            "mid_layer" : nmt.params,
            "optime" : optimizer.state_dict()
        }, "check.pt")

torch.save({
    "params_enc" : nmt.encoder.RNN.params,
    "params_dec" : nmt.decoder.RNN.params,
    "mid_layer" : nmt.params,
    "optime" : optimizer.state_dict()
}, "check.pt")


# The constituent of the sentence (either words or letters should be a part of word_to_dict)
# Choose the words carefully!

str = "<START> "
start_string = "resumption of the session"
start_string = str + start_string + " <END>"

words = start_string.split()
num_len_start = len(words)
word_enc = [word_to_idx_en[word] for word in words]
word_enc = torch.tensor(word_enc).unsqueeze(0)

hidden_enc_states, prev_h, mask = nmt.encoder(word_enc) # (1, hidden_state_dim)
prev_h = prev_h @ nmt.params["W_mid"] + nmt.params["b_mid"]
prev_c = torch.zeros((1, prev_h.shape[1])) 

start_weights = nmt.decoder.RNN.params['W_embed'][nmt.decoder.RNN.start_token]
start_weights = torch.unsqueeze(start_weights, 0) # (1, word_vec_dim)

curr_x = start_weights

next_h, next_c = None, None
max_length = 60

rnn = nmt.decoder.RNN
captions = rnn.null_token * torch.ones((1, max_length), dtype=torch.int32)

letter_or_word = words

for i in range(max_length):
    context = attention_context_vector(prev_h, hidden_enc_states, rnn.params["W_attention1"],\
                                rnn.params["b_attention1"], rnn.params["W_attention2"], mask)
    if(architecture == "rnn"):
        next_h = rnn_attention_step_forward(curr_x, prev_h, rnn.params["Wx"], rnn.params["Wh"],\
                                            rnn.params["b"], context, rnn.params["W_context"])
    else:
        next_h, next_c = lstm_attention_step_forward(curr_x, prev_h, prev_c, rnn.params["Wx"], rnn.params["Wh"],\
                                                    rnn.params["b"], context, rnn.params["W_context"])

    out = affine_forward(next_h, rnn.params["W_vocab"], rnn.params["b_vocab"])
    T = 0.5
    out = torch.exp(out/T)
    out = out / torch.sum(out, dim = 1)
    indices = torch.multinomial(out, 1).squeeze(0)
    captions[0, i] = indices
    prev_h = next_h
    prev_c = next_c
    curr_x = rnn.params["W_embed"][indices]

captions = captions.tolist()

file = open("out.txt",'w')

for i in range(len(captions)):
    words = [reverse_dict_es[val] for val in captions[i]]
    for word in words:
        file.write(word + " ")
        print(word + " ", end = "")
    print("******")