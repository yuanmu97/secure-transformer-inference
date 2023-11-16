"""
reference: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
"""
import torch 
import torch.nn as nn
import math
import numpy as np
import copy
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)


    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size() 
        # d_model = num_heads * d_k
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) 
    

    def scaled_dot_product_attention(self, q, k, v, mask):
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) 
        # shape [BS, NUM_HEADS, SEQ_LEN, SEQ_LEN]

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_probs = torch.softmax(attention_scores, dim=-1)

        return torch.matmul(attention_probs, v)
    
    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
         
    
    def forward(self, q, k, v, mask):
        q = self.split_heads(self.w_q(q)) 
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))

        attention_output = self.scaled_dot_product_attention(q, k, v, mask)
        attention_output = self.combine_heads(attention_output)

        return self.w_o(attention_output) 


class FeedForward(nn.Module):
    """
    """
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class EncoderBlock(nn.Module):
    """
    """
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderBlock, self).__init__()
        self.attention_layer = MultiHeadAttention(d_model, num_heads)
        self.ff_layer = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attention_output = self.attention_layer(x, x, x, mask)
        x0 = x + attention_output
        x1 = self.ln1(x0)

        ff_output = self.ff_layer(x1)
        x2 = self.ln2(x1 + ff_output)
        return x2


class DecoderBlock(nn.Module):
    """
    """
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ff_layer = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        self_attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.ln1(x + self_attention_output)
        cross_attention_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.ln2(x + cross_attention_output)
        ff_output = self.ff_layer(x)
        x = self.ln3(x + ff_output)
        return x
    

class Transformer(nn.Module):
    """
    """
    def __init__(self, d_model, num_heads, d_ff, num_layers, output_dim):
        super(Transformer, self).__init__()
        self.enc_layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, output_dim)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, x, target):
        src_mask, tgt_mask = self.generate_mask(x, target)

        enc_output = x
        for enc_l in self.enc_layers:
            enc_output = enc_l(enc_output, src_mask)

        dec_output = target
        for dec_l in self.dec_layers:
            dec_output = dec_l(dec_output, enc_output, src_mask, tgt_mask)
        
        output = self.fc(dec_output)
        return output


def inv_permutation(p):
    inv_p = [0]*len(p)
    for old_idx, new_idx in enumerate(p):
        inv_p[new_idx] = old_idx
    return inv_p


def permute_attention(attn_layer, p):
    inv_p = inv_permutation(p)
    print(inv_p)
    p_attn_layer = copy.deepcopy(attn_layer)
    with torch.no_grad():
        for name, para in p_attn_layer.named_parameters():
            if name in ["w_q.weight", "w_k.weight", "w_v.weight"]:
                para.data = para.data[:, p] 
            if name in ["w_o.weight", "w_o.bias"]:
                para.data = para.data[p]
    return p_attn_layer


def permute_feedforward(ffn_layer, p):
    inv_p = inv_permutation(p)
    p_ffn_layer = copy.deepcopy(ffn_layer)
    with torch.no_grad():
        for name, para in p_ffn_layer.named_parameters():
            # print(name)
            if name in ["fc1.weight"]:
                para.data = para.data[:, p]
            if name in ["fc2.weight", "fc2.bias"]:
                para.data = para.data[p]
    return p_ffn_layer


def permute_block(blk, p):
    inv_p = inv_permutation(p)
    p_blk = copy.deepcopy(blk)
    with torch.no_grad():
        for name, para in p_blk.named_parameters():
            if name in ["attention_layer.w_q.weight",
                        "attention_layer.w_k.weight",
                        "attention_layer.w_v.weight",
                        "ff_layer.fc1.weight"]:
                para.data = para.data[:, p]
            if name in ["attention_layer.w_o.weight",
                        "attention_layer.w_o.bias",
                        "ff_layer.fc2.weight",
                        "ff_layer.fc2.bias"]:
                para.data = para.data[p]

    return p_blk


if __name__ == "__main__":
    BS = 2
    SEQLEN = 3
    DMODEL = 4
    DFF = 8
    NHEADS = 2

    TEST_ATT = 0
    TEST_FFN = 0
    TEST_BLK = 1

    if TEST_ATT:
        x = torch.from_numpy(np.random.rand(BS, SEQLEN, DMODEL)).float()

        attention_layer = MultiHeadAttention(d_model=DMODEL, num_heads=NHEADS)
        mask = (1 - torch.triu(torch.ones(1, SEQLEN, SEQLEN), diagonal=1)).bool()
        with torch.no_grad():
            y = attention_layer(x, x, x, None)
            y_mask = attention_layer(x, x, x, mask)

        p = np.random.permutation(x.shape[2])
        print(p)

        xp = x[:, :, p]
        p_attention_layer = permute_attention(attention_layer, p)

        with torch.no_grad():
            print("Without mask:")
            yp = p_attention_layer(xp, xp, xp, None)
            diff = np.abs(y[:, :, p] - yp).sum()
            print("Original reslut:\n", y, "\nNew result:\n", yp)
            print("Diff=", diff)

            print("With mask:")
            yp_mask = p_attention_layer(xp, xp, xp, mask)
            diff_mask = np.abs(y_mask[:, :, p] - yp_mask).sum()
            print("Original reslut:\n", y_mask, "\nNew result:\n", yp_mask)
            print("Diff=", diff_mask)

    if TEST_FFN:
        x = torch.from_numpy(np.random.rand(BS, SEQLEN, DMODEL)).float()
        ffn_layer = FeedForward(DMODEL, DFF)

        p = np.random.permutation(DMODEL)
        print(f"Permutation={p}")

        xp = x[:, :, p]
        p_ffn_layer = permute_feedforward(ffn_layer, p)

        with torch.no_grad():
            print("Feedforward Layer:")
            y = ffn_layer(x)
            yp = p_ffn_layer(xp)
            diff = np.abs(y[:, :, p] - yp).sum()
            print("Original reslut:\n", y, "\nNew result:\n", yp)
            print("Diff=", diff)

    
    if TEST_BLK:
        x = torch.from_numpy(np.random.rand(BS, SEQLEN, DMODEL)).float()
        block = EncoderBlock(DMODEL, NHEADS, DFF)
        p = np.random.permutation(DMODEL)

        print(f"Permutation={p}")
        xp = x[:, :, p]
        p_block = permute_block(block, p)
        mask = (1 - torch.triu(torch.ones(1, SEQLEN, SEQLEN), diagonal=1)).bool()
        
        with torch.no_grad():
            print("Encoder Block:")
            y = block(x, None)
            yp = p_block(xp, None)
            diff = np.abs(y[:, :, p] - yp).sum()
            print("Original reslut:\n", y, "\nNew result:\n", yp)
            print("Diff=", diff)

            print("Decoder Block:")
            y = block(x, mask)
            yp = p_block(xp, mask)
            diff = np.abs(y[:, :, p] - yp).sum()
            print("Original reslut:\n", y, "\nNew result:\n", yp)
            print("Diff=", diff)
