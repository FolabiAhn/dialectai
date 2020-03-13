import torch
import torch.nn as nn
import torch.nn.functional as F




def smoothing(x):
    """Compute softmax values for each sets of scores in x."""
    return x.sigmoid() / x.sigmoid().sum(0)


class BahdanauAttentionBase(nn.Module):
    
    def __init__(self, units, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, units)
        self.W2 = nn.Linear(hidden_size, units)
        self.V = nn.Linear(units, 1)
        
    def forward(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        query = torch.squeeze(query, 0)
        hidden_with_time_axis = torch.unsqueeze(query, 1)
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        sum_1 = self.W1(values) + self.W2(hidden_with_time_axis)
        score = self.V(torch.tanh(sum_1))
        # score shape == (batch_size, max_length, 1)
        #attention_weights = F.softmax(score, dim=1)
        attention_weights = smoothing(score)
        # context_vector shape after sum == (batch_size, hidden_size) 
        # (values == EO)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights, score        
        
        
class BahdanauAttentionAudio(nn.Module):
    
    def __init__(self, kernel_size, units, hidden_size):
        super().__init__()
        self.prev_att = torch.zeros(20, 198, 1).to('cuda')#None
        self.W1 = nn.Linear(hidden_size, units)
        self.W2 = nn.Linear(hidden_size, units)
        self.V = nn.Linear(units, 1)
        self.loc_conv = nn.Conv1d(in_channels=198, out_channels=198, kernel_size=2*kernel_size+1,
                                  padding=kernel_size, bias=False)
        self.loc_proj = nn.Linear(1, hidden_size, bias=False)
 
    #def reset_mem(self):
    #    super().reset_mem()
    #    self.prev_att = None

    #def set_mem(self, prev_att):
    #    self.prev_att = prev_att
        
    def forward(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        query = torch.squeeze(query, 0)
        hidden_with_time_axis = torch.unsqueeze(query, 1)

        # Calculate location context
        convo = self.loc_conv(self.prev_att)
        loc_context = self.loc_proj(convo)
        
        #print("loc_context", loc_context.shape)
        
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        sum_1 = self.W1(values) + self.W2(hidden_with_time_axis) + loc_context
        
        #print("sum_1", sum_1.shape)
        score = self.V(torch.tanh(sum_1))
        # As we are dealing with audio we will take the topk frames
        top_val, top_pos = torch.topk(score, k=100, dim=1)
        score = score.squeeze(2)
        top_pos = top_pos.squeeze(2)
        score = -1 * (score.scatter(1, top_pos, 0) - score)
        score = score.unsqueeze(2)
        # We will change the softmax with the smoothing
        attention_weights = smoothing(score)
        self.prev_att = attention_weights
        
        # context_vector shape after sum == (batch_size, hidden_size) 
        # (values == EO)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights, score  
                      

            
class LuongAttentionDot(nn.Module):

    def __init__(self):
        super().__init__()
        
    def forward(self, query, values): # h_t = query      h_s = values
        query = torch.squeeze(query, 0)
        query = torch.unsqueeze(query, 1)
        query_transposed = query.transpose(2, 1)  
        score = torch.matmul(values, query_transposed) 
        attention_weights = F.softmax(score, dim=1)
        # context_vector shape after sum == (batch_size, hidden_size) 
        # (values == EO)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights, score


class LuongAttentionGeneral(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)

        
    def forward(self, query, values):
        query = torch.squeeze(query, 0)
        query = torch.unsqueeze(query, 1)
        query_transposed = query.transpose(2, 1) 
        score = torch.matmul(self.W(values), query_transposed)     
        # score shape == (batch_size, max_length, 1)
        attention_weights = F.softmax(score, dim=1)
        # context_vector shape after sum == (batch_size, hidden_size) 
        # (values == EO)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights, score


class LuongAttentionConcat(nn.Module):
    
    def __init__(self, units, hidden_size):
        super().__init__()
        self.W = nn.Linear(2 * hidden_size, units)
        self.V = nn.Linear(units, 1)
        
    def forward(self, query, values):
        query = torch.squeeze(query, 0)
        query = torch.unsqueeze(query, 1)
        query = query.repeat(1, values.shape[1], 1)
        
        cat = torch.cat((values, query), dim=2)
        score = self.V(torch.tanh(self.W(cat)))
        # score shape == (batch_size, max_length, 1)
        attention_weights = F.softmax(score, dim=1)
        # context_vector shape after sum == (batch_size, hidden_size) 
        # (values == EO)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights, score

        

class SuperHeadAttention(nn.Module):
    
    def __init__(self, units, hidden_size):
        super().__init__()
        self.nbr_heads = 8
        self.W = nn.Linear(self.nbr_heads, 1)
        self.attention_1 = BahdanauAttentionBase(units=units, hidden_size=hidden_size)
        self.attention_2 = BahdanauAttentionBase(units=units, hidden_size=hidden_size)
        self.attention_3 = BahdanauAttentionBase(units=units, hidden_size=hidden_size)
        self.attention_4 = BahdanauAttentionBase(units=units, hidden_size=hidden_size)
        self.attention_5 = BahdanauAttentionBase(units=units, hidden_size=hidden_size)
        self.attention_6 = BahdanauAttentionBase(units=units, hidden_size=hidden_size)
        self.attention_7 = BahdanauAttentionBase(units=units, hidden_size=hidden_size)
        self.attention_8 = BahdanauAttentionBase(units=units, hidden_size=hidden_size)
        
    def forward(self, query, values):
        
        _, _, score_1 = self.attention_1(query, values)
        _, _, score_2 = self.attention_2(query, values)
        _, _, score_3 = self.attention_3(query, values)
        _, _, score_4 = self.attention_4(query, values)
        _, _, score_5 = self.attention_5(query, values)
        _, _, score_6 = self.attention_6(query, values)
        _, _, score_7 = self.attention_7(query, values)
        _, _, score_8 = self.attention_8(query, values)
        
        concat = torch.cat([score_1, score_2, score_3, score_4,\
                            score_5, score_6, score_7, score_8], dim=2)
        #print('tata',concat.shape)
        score = self.W(concat)
        # As we are dealing with audio we will take the topk frames
        top_val, top_pos = torch.topk(score, k=100, dim=1)
        score = score.squeeze(2)
        top_pos = top_pos.squeeze(2)
        score = -1 * (score.scatter(1, top_pos, 0) - score)
        score = score.unsqueeze(2)
        # We will change the softmax with the smothing
        
        attention_weights = smoothing(score)
        #print('toto', score.shape)
        #attention_weights = F.softmax(score, dim=1)
        # context_vector shape after sum == (batch_size, hidden_size) 
        # (values == EO)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights, score
        
        
        