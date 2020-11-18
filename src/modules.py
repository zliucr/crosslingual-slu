
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from src.data_reader import PAD_INDEX, SLOT_PAD_INDEX
from src.utils import load_embedding

SLOT_PAD = 0

class Lstm(nn.Module):
    def __init__(self, params, vocab_en, vocab_trans):
        super(Lstm, self).__init__()
        self.n_layer = params.n_layer
        self.vocab_en = vocab_en
        self.vocab_trans = vocab_trans
        self.emb_dim = params.emb_dim
        self.hidden_dim = params.hidden_dim
        self.dropout = params.dropout
        self.bidirection = params.bidirection
        self.embnoise = params.embnoise
        self.emb_file_en = params.emb_file_en
        self.emb_file_trans = params.emb_file_trans

        if params.tar_only == False or params.zs == True:
            # embedding layer
            self.embedding_en = nn.Embedding(self.vocab_en.n_words, self.emb_dim, padding_idx=PAD_INDEX)
            # load embedding
            embedding_en = load_embedding(vocab_en, self.emb_dim, self.emb_file_en)
            self.embedding_en.weight.data.copy_(torch.FloatTensor(embedding_en))

        self.embedding_trans = nn.Embedding(self.vocab_trans.n_words, self.emb_dim, padding_idx=PAD_INDEX)
        # load embedding
        embedding_trans = load_embedding(vocab_trans, self.emb_dim, self.emb_file_trans)
        self.embedding_trans.weight.data.copy_(torch.FloatTensor(embedding_trans))

        # LSTM layers
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.n_layer, 
                        dropout=self.dropout, bidirectional=self.bidirection, batch_first=True)
    
    def forward(self, X, input_type):
        """
        Input:
            x: text (bsz, seq_len)
            input_type: either "en" or "trans"
        Output:
            last_layer: last layer of lstm (bsz, seq_len, hidden_dim)
        """
        if input_type == "en":
            embeddings = self.embedding_en(X)
        else:
            embeddings = self.embedding_trans(X)
        embeddings = embeddings.detach()
        if self.embnoise == True and self.training == True:
            # add embedding noise
            size = embeddings.size()
            noise = torch.randn(size) * 0.05
            noise = noise.cuda()
            embeddings += noise
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)

        # LSTM
        # lstm_output (batch_first): (bsz, seq_len, hidden_dim)
        lstm_output, (_, _) = self.lstm(embeddings)

        return lstm_output

class Lstm4pretr(nn.Module):
    def __init__(self, params, vocab_en):
        super(Lstm4pretr, self).__init__()
        self.n_layer = params.n_layer
        self.n_words_en = vocab_en.n_words
        self.emb_dim = params.emb_dim
        self.hidden_dim = params.hidden_dim
        self.dropout = params.dropout
        self.bidirection = params.bidirection
        self.embnoise = params.embnoise
        self.emb_file_en = params.emb_file_en

        # embedding layer
        self.embedding_en = nn.Embedding(self.n_words_en, self.emb_dim, padding_idx=PAD_INDEX)
        # load embedding
        embedding_en = load_embedding(vocab_en, self.emb_dim, self.emb_file_en)
        self.embedding_en.weight.data.copy_(torch.FloatTensor(embedding_en))

        # LSTM layers
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.n_layer, 
                        dropout=self.dropout, bidirectional=self.bidirection, batch_first=True)
    
    def forward(self, X):
        """
        Input:
            x: text (bsz, seq_len)
        Output:
            last_layer: last layer of lstm (bsz, seq_len, hidden_dim)
        """
        embeddings = self.embedding_en(X)
        embeddings = embeddings.detach()
        if self.embnoise == True and self.training == True:
            # add embedding noise
            size = embeddings.size()
            noise = torch.randn(size) * 0.05
            noise = noise.cuda()
            embeddings += noise
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)

        # LSTM
        # lstm_output (batch_first): (bsz, seq_len, hidden_dim)
        lstm_output, (_, _) = self.lstm(embeddings)

        return lstm_output

# https://github.com/huggingface/torchMoji/blob/master/torchmoji/attlayer.py
class Attention(nn.Module):
    """
    Computes a weighted average of channels across timesteps (1 parameter pr. channel).
    """
    def __init__(self, attention_size, return_attention=False):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
            return_attention: If true, output will include the weight for each input token
                              used for the prediction
        """
        super(Attention, self).__init__()
        self.return_attention = return_attention
        self.attention_size = attention_size
        self.attention_vector = Parameter(torch.FloatTensor(attention_size))
        torch.nn.init.uniform(self.attention_vector.data, -0.01, 0.01)

    def __repr__(self):
        s = '{name}({attention_size}, return attention={return_attention})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        logits = inputs.matmul(self.attention_vector)
        unnorm_ai = (logits - logits.max()).exp()
        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)

        idxes = torch.arange(
            0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        idxes = idxes.cuda()
        mask = Variable((idxes < input_lengths.unsqueeze(1)).float())

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(
            inputs, attentions.unsqueeze(-1).expand_as(inputs))
        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)
        return (representations, attentions if self.return_attention else None)


class CRF(nn.Module):
    """
    Implements Conditional Random Fields that can be trained via
    backpropagation. 
    """
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(num_tags))

        nn.init.xavier_normal_(self.transitions)

    def forward(self, feats):
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        return self._viterbi(feats)

    def loss(self, feats, tags):
        """
        Computes negative log likelihood between features and tags.
        Essentially difference between individual sequence scores and 
        sum of all possible sequence scores (partition function)
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns:
            Negative log likelihood [a scalar] 
        """
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        if len(tags.shape) != 2:
            raise ValueError('tags must be 2-d but got {}-d'.format(tags.shape))

        if feats.shape[:2] != tags.shape:
            raise ValueError('First two dimensions of feats and tags must match ', feats.shape, tags.shape)

        sequence_score = self._sequence_score(feats, tags)
        partition_function = self._partition_function(feats)
        log_probability = sequence_score - partition_function

        # -ve of l()
        # Average across batch
        return -log_probability.mean()

    def _sequence_score(self, feats, tags):
        """
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns: Sequence score of shape [batch size]
        """

        batch_size = feats.shape[0]

        # Compute feature scores
        feat_score = feats.gather(2, tags.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        # print(feat_score.size())

        # Compute transition scores
        # Unfold to get [from, to] tag index pairs
        tags_pairs = tags.unfold(1, 2, 1)

        # Use advanced indexing to pull out required transition scores
        indices = tags_pairs.permute(2, 0, 1).chunk(2)
        trans_score = self.transitions[indices].squeeze(0).sum(dim=-1)

        # Compute start and stop scores
        start_score = self.start_transitions[tags[:, 0]]
        stop_score = self.stop_transitions[tags[:, -1]]

        return feat_score + start_score + trans_score + stop_score

    def _partition_function(self, feats):
        """
        Computes the partitition function for CRF using the forward algorithm.
        Basically calculate scores for all possible tag sequences for 
        the given feature vector sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns:
            Total scores of shape [batch size]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        a = feats[:, 0] + self.start_transitions.unsqueeze(0) # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(0) # [1, num_tags, num_tags] from -> to

        for i in range(1, seq_size):
            feat = feats[:, i].unsqueeze(1) # [batch_size, 1, num_tags]
            a = self._log_sum_exp(a.unsqueeze(-1) + transitions + feat, 1) # [batch_size, num_tags]

        return self._log_sum_exp(a + self.stop_transitions.unsqueeze(0), 1) # [batch_size]

    def _viterbi(self, feats):
        """
        Uses Viterbi algorithm to predict the best sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns: Best tag sequence [batch size, sequence length]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))
        
        v = feats[:, 0] + self.start_transitions.unsqueeze(0) # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(0) # [1, num_tags, num_tags] from -> to
        paths = []

        for i in range(1, seq_size):
            feat = feats[:, i] # [batch_size, num_tags]
            v, idx = (v.unsqueeze(-1) + transitions).max(1) # [batch_size, num_tags], [batch_size, num_tags]
            
            paths.append(idx)
            v = (v + feat) # [batch_size, num_tags]

        
        v, tag = (v + self.stop_transitions.unsqueeze(0)).max(1, True)

        # Backtrack
        tags = [tag]
        for idx in reversed(paths):
            tag = idx.gather(1, tag)
            tags.append(tag)

        tags.reverse()
        return torch.cat(tags, 1)

    
    def _log_sum_exp(self, logits, dim):
        """
        Computes log-sum-exp in a stable way
        """
        max_val, _ = logits.max(dim)
        return max_val + (logits - max_val.unsqueeze(dim)).exp().sum(dim).log()


class IntentPredictor(nn.Module):
    def __init__(self, params):
        super(IntentPredictor, self).__init__()
        self.label_reg = params.la_reg
        self.lvm = params.lvm
        self.num_intent = params.num_intent
        self.intent_adv = params.intent_adv
        self.attention_size = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.atten_layer = Attention(attention_size=self.attention_size, return_attention=False)

        if self.lvm == False:
            self.linear = nn.Linear(self.attention_size, self.num_intent)
        else:
            self.lvm_dim = params.lvm_dim
            self.mean = nn.Linear(self.attention_size, self.lvm_dim)
            self.var = nn.Linear(self.attention_size, self.lvm_dim)
            self.linear = nn.Linear(self.lvm_dim, self.num_intent)

    def forward(self, inputs, lengths):
        """ forward pass
        Inputs:
            inputs: lstm hidden layer (bsz, seq_len, hidden_dim)
            lengths: lengths of x (bsz, )
        Output:
            prediction: Intent prediction (bsz, num_intent)
        """
        atten_layer, _ = self.atten_layer(inputs, lengths)
        if self.lvm == False:
            prediction = self.linear(atten_layer)
        else:
            if self.training == True:
                mean = self.mean(atten_layer)
                var =  self.var(atten_layer)
                
                size = var.size()
                noise = torch.randn(size)
                noise = noise.cuda()
                z = mean + torch.exp(var / 2) * noise * 0.5
            else:
                z = self.mean(atten_layer)
            prediction = self.linear(z)
        if self.label_reg == False or self.training == False:
            if self.intent_adv == True:
                return prediction, z
            else:
                return prediction
        else:
            if self.intent_adv == True:
                return prediction, atten_layer, z
            else:
                return prediction, atten_layer


class SlotPredictor(nn.Module):
    def __init__(self, params):
        super(SlotPredictor, self).__init__()
        self.lvm = params.lvm
        self.adversarial = params.adv
        self.num_slot = params.num_slot
        self.hidden_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim

        if self.lvm == False:
            self.linear = nn.Linear(self.hidden_dim, self.num_slot)
        else:
            self.lvm_dim = params.lvm_dim
            self.mean = nn.Linear(self.hidden_dim, self.lvm_dim)
            self.var = nn.Linear(self.hidden_dim, self.lvm_dim)
            self.linear = nn.Linear(self.lvm_dim, self.num_slot)

        self.crf_layer = CRF(self.num_slot)

    def forward(self, inputs):
        """ forward pass
        Input:
            inputs: lstm hidden layer (bsz, seq_len, hidden_dim)
        Output:
            prediction: slot prediction (bsz, seq_len, num_slot)
        """
        if self.lvm == False:
            prediction = self.linear(inputs)
            return prediction
        else:
            if self.training == True:
                mean = self.mean(inputs)
                var = self.var(inputs)
                size = var.size()
                noise = torch.randn(size)
                noise = noise.cuda()
                z = mean + torch.exp(var / 2) * noise * 0.1
            else:
                z = self.mean(inputs)
            prediction = self.linear(z)
            if self.adversarial == True and self.training == True:
                return prediction, z
            else:
                return prediction
    
    def gen_latent_variables(self, inputs):
        mean = self.mean(inputs)
        var = self.var(inputs)
        size = var.size()
        noise = torch.randn(size)
        noise = noise.cuda()
        z = mean + torch.exp(var / 2) * noise * 0.1

        return z

    def crf_loss(self, inputs, lengths, y):
        """ create crf loss
        Input:
            inputs: output of SlotPredictor (bsz, seq_len, num_slot)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        padded_y = self.pad_label(lengths, y)
        crf_loss = self.crf_layer.loss(inputs, padded_y)

        return crf_loss

    def decode(self, prediction, lengths):
        prediction = [ prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths) ]
        return prediction

    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: output of SlotPredictor (bsz, seq_len, num_slot)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.crf_layer(inputs)
        prediction = [ prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths) ]

        return prediction

    def make_mask(self, lengths):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        mask = torch.LongTensor(bsz, max_len).fill_(1)
        for i in range(bsz):
            length = lengths[i]
            mask[i, length:max_len] = 0
        mask = mask.cuda()
        return mask
    
    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y


class LabelEncoder(nn.Module):
    def __init__(self, params):
        super(LabelEncoder, self).__init__()

        self.num_label = params.num_slot + 1   # include padding
        self.n_layer = params.n_layer_la_enc
        self.emb_dim = params.emb_dim_la_enc
        self.hidden_dim = params.hidden_dim_la_enc
        self.bidirection = params.bidirection
        self.dropout = params.dropout

        # embeddings
        self.embedding = nn.Embedding(self.num_label, self.emb_dim, padding_idx=SLOT_PAD_INDEX)
        # LSTM layers
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.n_layer, 
                        dropout=self.dropout, bidirectional=self.bidirection, batch_first=True)
        
        # Attention layer
        self.attention_size = self.hidden_dim * 2 if self.bidirection else self.hidden_dim
        self.atten_layer = Attention(attention_size=self.attention_size, return_attention=False)
    
    def forward(self, X, lengths):
        embeddings = self.embedding(X)
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        lstm_hidden, (_, _) = self.lstm(embeddings)

        label_repre, _ = self.atten_layer(lstm_hidden, lengths)

        return label_repre


class UniformDistriGen(nn.Module):
    def __init__(self, params):
        super(UniformDistriGen, self).__init__()
        self.lvm_dim = params.lvm_dim
        self.num_slot = params.num_slot
        self.uniform_distri_genertor = nn.Linear(self.lvm_dim, self.num_slot+1)  # +1 for SLOT_PAD_INDEX
        self.intent_adv = params.intent_adv
        if self.intent_adv == True:
            self.num_intent = params.num_intent
            # uniform distribution generator for intent
            self.udg4intent = nn.Linear(self.lvm_dim, self.num_intent)
    
    def forward(self, feats, feats_intent=None):
        out = self.uniform_distri_genertor(feats)
        if self.intent_adv == True:
            out_intent = self.udg4intent(feats_intent)
            return out, out_intent
        else:
            return out
