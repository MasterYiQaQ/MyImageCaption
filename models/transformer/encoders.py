from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention,MyMultiHeadAttention
from models.transformer.geom import ratio_geomertical
from torch.autograd import Variable
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module, attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        outs = []
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs, attention_mask


class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        return super(MemoryAugmentedEncoder, self).forward(out, attention_weights=attention_weights)

class VisualPlusSemeticEncoder(nn.Module):
    def __init__(self,vocab_size,N, padding_idx,identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(VisualPlusSemeticEncoder, self).__init__()
        self.d_model = 512
        self.h = 8
        self.d_k = 64
        self.d_v = 64
        self.d_ff = 2048
        self.dropout_num = .1
        self.vocab_size = vocab_size
        self.layer_norm_box = nn.LayerNorm(64)
        self.layer_norm_model = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=self.dropout_num)
        self.word_multi = MultiHeadAttention(self.d_model,self.d_k,self.d_v,self.h)
        self.visual_multi = MultiHeadAttention(self.d_model, self.d_k, self.d_v, self.h)
        # self.ffd = PositionWiseFeedForward(self.d_model*2,self.d_ff)
        self.fc = nn.Linear(2048,512)
        self.linear1 = nn.Linear(4,64)
        self.linear2 = nn.Linear(64 + 512, 512)
        self.linear3 = nn.Linear(1024,512)

        self.word_emb = nn.Embedding(vocab_size, self.d_model, padding_idx=padding_idx)

        self.Encoder = MultiLevelEncoder(N,padding_idx,attention_module=attention_module, attention_module_kwargs=attention_module_kwargs)

        self.semestic_attention = MyMultiHeadAttention(self.d_model,self.d_k,self.d_v,self.h,self.dropout_num, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module, attention_module_kwargs=attention_module_kwargs)
        self.visual_attention = MyMultiHeadAttention(self.d_model,self.d_k,self.d_v,self.h,self.dropout_num, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module, attention_module_kwargs=attention_module_kwargs)

    def forward(self,images,bbox,label,attention_weights=None):

        semestic = F.relu(self.word_emb(label.long()))
        # semestic = self.dropout(semestic)
        # semestic = self.layer_norm_model(semestic)


        box = F.relu(self.linear1(bbox))
        box = self.dropout(box)
        box = self.layer_norm_box(box)

        image = F.relu(self.fc(images))
        image = self.dropout(image)
        image = self.layer_norm_model(image)

        visual = torch.cat((image,box),2)
        visual = F.relu(self.linear2(visual))
        visual = self.dropout(visual)
        visual = self.layer_norm_model(visual)

        #visualandsemestic = torch.cat((image,semestic),2)
        #SGA
        s_v = self.semestic_attention(semestic,visual,visual)
        #VGA
        print()
        v_s = self.visual_attention(visual,semestic,semestic)
        #
        feature = torch.cat((s_v,v_s),2)
        feature = self.linear3(feature)
        feature = self.dropout(feature)
        feature = self.layer_norm_model(feature)

        out,attention = self.Encoder(feature,attention_weights=attention_weights)

        return out,attention