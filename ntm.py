import torch
import torch.nn as nn
from ntm_data import *

class NTM(nn.Module):
    def __init__(self, hidden_state_dim, input_dim, num_memory_cells, memory_cell_dim):
        super(NTM, self).__init__()
    
        self.num_memory_cells = num_memory_cells
        self.memory_cell_dim = memory_cell_dim

        # Declaring only 1 read head and only 1 write head for now
        self.read_head = ReadHead(hidden_state_dim, memory_cell_dim, num_memory_cells)
        self.write_head = WriteHead(hidden_state_dim, memory_cell_dim, num_memory_cells)
        self.read_weights_init = torch.rand(num_memory_cells)
        self.write_weights_init = torch.rand(num_memory_cells)
        
        self.h_init = torch.nn.Parameter(torch.randn(hidden_state_dim))
        # RNN internal layers that computes the affine transform which maps the input and the hidden state
        # to the next state

        # GRU unit
        self.rnn_wx_update = torch.nn.Linear(input_dim, hidden_state_dim)
        self.rnn_wh_update = torch.nn.Linear(hidden_state_dim, hidden_state_dim)

        self.rnn_wx_reset = torch.nn.Linear(input_dim, hidden_state_dim)
        self.rnn_wh_reset = torch.nn.Linear(hidden_state_dim, hidden_state_dim)

        self.rnn_wx_hidden = torch.nn.Linear(input_dim, hidden_state_dim)
        self.rnn_wh_hidden = torch.nn.Linear(hidden_state_dim, hidden_state_dim)

        # Secondary state change based on read vector
        self.rnn_read_secondary_effect = torch.nn.Linear(memory_cell_dim, hidden_state_dim)
        self.rnn_self_secondary_effect = torch.nn.Linear(hidden_state_dim, hidden_state_dim)

        # Define the ouptut layer depending on the use case
        self.output_affine = torch.nn.Linear(hidden_state_dim, input_dim)

    def forward(self, input, output):
        # Assuming the input and output to be of the shape (N, T, W)
        # N -> Mini batch size, T -> The size of the temporal component, W -> The size of one such component

        N, T, W = input.shape
        prev_h = torch.tile(self.h_init, (N, 1))
        # Intialize the memory to zeros
        memory = torch.zeros(N, self.num_memory_cells, self.memory_cell_dim)
        prev_weights_read = torch.tile(self.read_weights_init, (N, 1))
        prev_weights_write = torch.tile(self.write_weights_init, (N, 1))
        loss = 0
        for i in range(T):
            z = torch.sigmoid(self.rnn_wx_update(input[:,i,:]) + self.rnn_wh_update(prev_h))
            r = torch.sigmoid(self.rnn_wx_reset(input[:,i,:]) + self.rnn_wh_reset(prev_h))
            prev_h = (1 - z) * prev_h + z * torch.tanh(self.rnn_wx_hidden(input[:,i,:]) + self.rnn_wh_hidden(prev_h) * r)
            # Pass the hidden state to the read head first, then the write head
            read, prev_weights_read = self.read_head(memory, prev_weights_read, prev_h)
            memory, prev_weights_write = self.write_head(memory, prev_weights_write, prev_h)
            # Use read to update the hidden state of the controller
            secondary_read_effect = self.rnn_read_secondary_effect(read)
            secondary_self_effect = self.rnn_self_secondary_effect(prev_h)
            prev_h = torch.tanh(secondary_read_effect + secondary_self_effect)
            # Based on this prev_h, calculate the output and subsequently the loss
            out = self.output_affine(prev_h)
            loss += torch.sum(((out - output[:,i,:]) ** 2))

        return loss

class ReadHead(nn.Module):
    def __init__(self, input_dim, memory_cell_dim, num_memory_cells):
        super(ReadHead, self).__init__()

        # Based on the hidden_dim of the state of the controller, computes 5 components

        # 1). k_t, the key vector which is a vector of dim same as that of any memory cell (memory_cell_dim)
        # 2). beta_t which is a scalar used to denote key strength for content based addressing
        # 3). g_t which is a scalar, which is used to selectively choose between the w_{t-1} and content based weights
        # 4). s_t which is the convolution shift vector used to give the probablity of each shift, of dimension num_memory_cells
        # 5). gamma_t which is a scalar used to define the precision of the final weights (used for removing blurriness)

        # The computation for the final weights depends upon the memory cells and the previous weights
        # Assume prev_weights are given by the caller in the forward function

        # Defining 5 hidden layers of appropriate dimensions

        self.k_affine = nn.Linear(input_dim, memory_cell_dim)
        self.beta_affine = nn.Linear(input_dim, 1)
        self.g_affine = nn.Linear(input_dim, 1)
        self.s_affine = nn.Linear(input_dim, num_memory_cells)
        self.gamma_affine = nn.Linear(input_dim, 1)

    def forward(self, memory, prev_weights, controller_hidden_state):
        
        # Assume memory of shape (N, num_memory_cells, memory_cell_dim)
        # prev_weights of shape (N, num_memory_cells)
        # controller_hidden_state of shape (N, H)
        # This function will be called at each time step (0 ... T-1)
        # returns the read vector and the new weights

        # Use relu for different non-linearities for now
        N, H = prev_weights.shape
        N, num_memory_cells, memory_cell_dim = memory.shape

        key = torch.relu(self.k_affine(controller_hidden_state)) # (N, memory_cell_dim)

        # Compute the cosine simmilarity between the key and all the memory cells
        # Cosine similarity is just the dot product between the 2 vectors
        cos_similarity = torch.zeros(N, num_memory_cells)
        for i in range(N):
            cos_similarity = key @ torch.transpose(memory[i,:,:], 0, 1)

        # Compute beta and multiply each component of the cos_similarity matrix by beta
        beta = self.beta_affine(controller_hidden_state)
        cos_similarity = cos_similarity * beta

        # wc is given by the softmax along dim = -1
        w_content = torch.softmax(cos_similarity, dim = -1)

        # calculate the g scalar
        g = self.g_affine(controller_hidden_state)

        # Interpolate among the prev_weights and the w_content based on the value of g
        w_g = g * w_content + (1 - g) * prev_weights

        # Compute the shift vector and apply softmax to it
        shift = torch.softmax(self.s_affine(controller_hidden_state), dim = -1)

        # Compute the circular convolution
        circ = torch.zeros((N, num_memory_cells))
        shift_other_way = torch.flip(shift, [1])
        cat = torch.cat([shift_other_way, shift_other_way], dim = 1)
        for i in range(num_memory_cells):
            circ[:,i] = torch.sum(cat[:,num_memory_cells - 1 - i:2*num_memory_cells - 1 - i] * w_g, dim = -1)
        
        # Calculate gamma and raise weach weight to gamma and normalize
        gamma = self.g_affine(controller_hidden_state)
        w_g = circ ** gamma
        w_next = w_g / torch.sum(w_g, dim = -1).unsqueeze(-1)

        # Read from the memory, now that we have calculated the weights
        read = torch.zeros((N, memory_cell_dim))
        for i in range(N):
            read[i,:] = w_next[i,:] @ memory[i,:,:]

        return read, w_next
    
class WriteHead(nn.Module):

    def __init__(self, input_dim, memory_cell_dim, num_memory_cells):
        super(WriteHead, self).__init__()

        # Based on the hidden_dim of the state of the controller, computes 5 components

        # 1). k_t, the key vector which is a vector of dim same as that of any memory cell (memory_cell_dim)
        # 2). beta_t which is a scalar used to denote key strength for content based addressing
        # 3). g_t which is a scalar, which is used to selectively choose between the w_{t-1} and content based weights
        # 4). s_t which is the convolution shift vector used to give the probablity of each shift, of dimension num_memory_cells
        # 5). gamma_t which is a scalar used to define the precision of the final weights (used for removing blurriness)
        # 6). The erase vector which is a vector of dimension memory_cell_dim
        # 7). The add vector which is a vector of dimension memory_cell_dim

        # The computation for the final weights depends upon the memory cells and the previous weights
        # Assume prev_weights are given by the caller in the forward function

        # Defining 7 hidden layers of appropriate dimensions

        self.k_affine = nn.Linear(input_dim, memory_cell_dim)
        self.beta_affine = nn.Linear(input_dim, 1)
        self.g_affine = nn.Linear(input_dim, 1)
        self.s_affine = nn.Linear(input_dim, num_memory_cells)
        self.gamma_affine = nn.Linear(input_dim, 1)

        self.erase_affine = nn.Linear(input_dim, memory_cell_dim)
        self.add_affine = nn.Linear(input_dim, memory_cell_dim)

    def forward(self, memory, prev_weights, controller_hidden_state):
        
        # Assume memory of shape (N, num_memory_cells, memory_cell_dim)
        # prev_weights of shape (N, num_memory_cells)
        # controller_hidden_state of shape (N, H)
        # This function will be called at each time step (0 ... T-1)
        # returns the read vector and the new weights
        # Use relu for different non-linearities for now
        N, H = prev_weights.shape
        N, num_memory_cells, memory_cell_dim = memory.shape

        key = torch.relu(self.k_affine(controller_hidden_state)) # (N, memory_cell_dim)

        # Compute the cosine simmilarity between the key and all the memory cells
        # Cosine similarity is just the dot product between the 2 vectors
        cos_similarity = torch.zeros(N, num_memory_cells)
        for i in range(N):
            cos_similarity = key @ torch.transpose(memory[i,:,:], 0, 1)

        # Compute beta and multiply each component of the cos_similarity matrix by beta
        beta = self.beta_affine(controller_hidden_state)
        cos_similarity = cos_similarity * beta

        # wc is given by the softmax along dim = -1
        w_content = torch.softmax(cos_similarity, dim = -1)

        # calculate the g scalar
        g = self.g_affine(controller_hidden_state)

        # Interpolate among the prev_weights and the w_content based on the value of g
        w_g = g * w_content + (1 - g) * prev_weights

        # Compute the shift vector and apply softmax to it
        shift = torch.softmax(self.s_affine(controller_hidden_state), dim = -1)

        # Compute the circular convolution
        circ = torch.zeros((N, num_memory_cells))
        shift_other_way = torch.flip(shift, [1])
        cat = torch.cat([shift_other_way, shift_other_way], dim = 1)
        for i in range(num_memory_cells):
            circ[:,i] = torch.sum(cat[:,num_memory_cells - 1 - i:2*num_memory_cells - 1 - i] * w_g, dim = -1)
        
        # Calculate gamma and raise each weight to gamma and normalize
        gamma = self.g_affine(controller_hidden_state)
        w_g = circ ** gamma
        w_next = w_g / torch.sum(w_g, dim = -1).unsqueeze(-1)

        # Change memory inplace
        # Calculate erase and add vectors
        erase = self.erase_affine(controller_hidden_state)
        add = self.add_affine(controller_hidden_state)
        write = torch.zeros((N, memory_cell_dim))
        new_memory = torch.zeros_like(memory)
        for i in range(N):
            to_erase = torch.ones(num_memory_cells, memory_cell_dim) - torch.reshape(w_next[i, :], (num_memory_cells, 1)) * torch.tile(erase[i], (num_memory_cells, 1))
            write = memory[i,:,:] * to_erase
            to_add = torch.reshape(w_next[i, :], (num_memory_cells, 1)) * torch.tile(add[i], (num_memory_cells, 1))
            new_memory[i,:,:] = write + to_add

        return new_memory, w_next


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Common Parameters For Both Encoder and Decoder
# TODO make the parameters different for both of them, define separately

hidden_state_dim = 128
input_dim = 8
num_memory_cells = 128
memory_cell_dim = 20
batch_size = 50
seq_len = 10

input, output = copy_task_data(batch_size, seq_len, input_dim)

# Create a NTM instance
ntm = NTM(hidden_state_dim, input_dim, num_memory_cells, memory_cell_dim)
for name, mod in ntm.named_children():
    print(name, mod)
epochs = 1000
learning_rate = 0.001

optimizer = torch.optim.Adam(ntm.parameters(), learning_rate)

for i in range(epochs):

    loss = ntm(input, output)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    for p in ntm.parameters():
        if(torch.isnan(p).any()):
            for param in ntm.parameters():
                print(param)
            break
    torch.nn.utils.clip_grad_norm_(ntm.parameters(), 10.0)

    optimizer.step()