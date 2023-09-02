"""This file defines layer types that are commonly used for recurrent neural networks.
"""

import torch


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = x.reshape(x.shape[0], -1) @ w + b
    return out


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """Run the forward pass for a single timestep of a vanilla RNN using a tanh activation function.

    The input data has dimension D, the hidden state has dimension H,
    and the minibatch is of size N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D)
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    a_next = b + prev_h @ Wh + x @ Wx
    next_h = torch.tanh(a_next)
    return next_h


def rnn_attention_step_forward( x, prev_h, Wx, Wh, b, c, Wc):
    """Attend over all the hidden state values of the encoder
        h -> hidden states of the input of the shape (N, Te, H)
        x -> current input of shape(N, W)
        prev_h -> previous hiddens state
    """
    a_next = b + prev_h @ Wh + x @ Wx + c @ Wc
    next_h = torch.tanh(a_next)
    return next_h

def attention_context_vector(prev_h, hidden_enc_states, W_atten, b_atten, v_atten, mask):
    enc_dec_state = torch.cat([hidden_enc_states, torch.unsqueeze(prev_h, 1).tile((1, hidden_enc_states.shape[1], 1))], dim=-1)
    # Shape (N, T, 2*H)
    # enc_dec_state = affine_forward(enc_dec_state, W_atten, b_atten)
    
    enc_dec_state = (enc_dec_state.reshape(-1, enc_dec_state.shape[2]) @ W_atten + b_atten)
    
    # enc_dec_state = affine_forward(enc_dec_state, v_atten, torch.zeros((1)))
    enc_dec_state = (torch.tanh(enc_dec_state) @ v_atten).reshape(hidden_enc_states.shape[0], hidden_enc_states.shape[1])
    enc_dec_state = torch.softmax(enc_dec_state, dim = -1)
    weight_matrix = enc_dec_state
    enc_dec_state = (enc_dec_state * mask).unsqueeze(-1)
    
    enc_dec_state = enc_dec_state * hidden_enc_states
    enc_dec_state = enc_dec_state.sum(dim = 1)
    return enc_dec_state, weight_matrix


def rnn_attention_forward(x, h0, Wx, Wh, b, hidden_enc_states, W_atten, b_atten, v_atten, mask, Wc):
    prev_h = h0
    N, T, H = x.shape[0], x.shape[1], h0.shape[1]
    h = torch.zeros((N, T, H))
    for i in range(T):
      c, _ = attention_context_vector(prev_h, hidden_enc_states, W_atten, b_atten, v_atten, mask)
      next_h = rnn_attention_step_forward(x[:,i,:], prev_h, Wx, Wh, b, c, Wc)
      prev_h = next_h
      h[:, i, :] = next_h
    return h

def lstm_attention_step_forward(x, prev_h, prev_c, Wx, Wh, b, c, Wc):
    next_h, next_c = None, None
    H = prev_h.shape[1]
    a = x @ Wx + prev_h @ Wh + c @ Wc + b # (N * 4H)
    # assume first H of a contain computations for input, next for forget, next for output, next for block 
    ai = torch.sigmoid(a[:,:H])
    af = torch.sigmoid(a[:,H:2*H])
    ao = torch.sigmoid(a[:,2*H:3*H])
    ag = torch.tanh(a[:,3*H:4*H])
    next_c = prev_c * af + ai * ag
    next_h = torch.tanh(next_c) * ao

    return next_h, next_c

def lstm_attention_forward(x, h0, Wx, Wh, b, hidden_enc_states, W_atten, b_atten, v_atten, mask, Wc):
    N, T, D = x.shape
    H = h0.shape[1]
    prev_h = h0
    prev_c = torch.zeros((N, H))
    h = torch.zeros((N, T, H))

    for i in range(T):
        c, _ = attention_context_vector(prev_h, hidden_enc_states, W_atten, b_atten, v_atten, mask)
        next_h, next_c = lstm_attention_step_forward(x[:,i,:], prev_h, prev_c, Wx, Wh, b, c, Wc)
        h[:,i,:] = next_h
        prev_h = next_h
        prev_c = next_c

    return h

def rnn_forward(x, h0, Wx, Wh, b):
    """Run a vanilla RNN forward on an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the RNN forward,
    we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D)
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H)
    - cache: Values needed in the backward pass
    """
    prev_h = h0
    N, T, H = x.shape[0], x.shape[1], h0.shape[1]
    h = torch.zeros((N, T, H))
    for i in range(T):
      next_h = rnn_step_forward(x[:,i,:], prev_h, Wx, Wh, b)
      prev_h = next_h
      h[:, i, :] = next_h
    return h

def word_embedding_forward(x, W):
    """Forward pass for word embeddings.
    
    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out = W[x[:,:]]
    return out


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c = None, None
    H = prev_h.shape[1]
    a = x @ Wx + prev_h @ Wh + b # (N * 4H)
    # assume first H of a contain computations for input, next for forget, next for output, next for block 
    ai = torch.sigmoid(a[:,:H])
    af = torch.sigmoid(a[:,H:2*H])
    ao = torch.sigmoid(a[:,2*H:3*H])
    ag = torch.tanh(a[:,3*H:4*H])
    next_c = prev_c * af + ai * ag
    next_h = torch.tanh(next_c) * ao

    return next_h, next_c

def lstm_forward(x, h0, Wx, Wh, b):
    """Forward pass for an LSTM over an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the LSTM forward,
    we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell state is set to zero.
    Also note that the cell state is not returned; it is an internal variable to the LSTM and is not
    accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """

    N, T, D = x.shape
    H = h0.shape[1]
    prev_h = h0
    prev_c = torch.zeros((N, H))
    h = torch.zeros((N, T, H))

    for i in range(T):
        next_h, next_c = lstm_step_forward(x[:,i,:], prev_h, prev_c, Wx, Wh, b)
        h[:,i,:] = next_h
        prev_h = next_h
        prev_c = next_c

    return h

def temporal_affine_forward(x, w, b):
    """Forward pass for a temporal affine layer.
    
    The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = (x.reshape(N * T, D) @ w).reshape(N, T, M) + b
    return out


def temporal_softmax_loss(x, y, mask):
    """A temporal version of softmax loss for use in RNNs.
    
    We assume that we are making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores for all vocabulary
    elements at all timesteps, and y gives the indices of the ground-truth element at each timestep.
    We use a cross-entropy loss at each timestep, summing the loss over all timesteps and averaging
    across the minibatch.

    As an additional complication, we may want to ignore the model output at some timesteps, since
    sequences of different length may have been combined into a minibatch and padded with NULL
    tokens. The optional mask argument tells us which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = torch.exp(x_flat - torch.max(x_flat, dim=1, keepdims=True)[0])
    probs = probs / torch.sum(probs, dim=1, keepdims=True)
    loss = -torch.sum(mask_flat * torch.log(probs[torch.arange(N * T), y_flat])) / N

    return loss