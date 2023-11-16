"""
reference: https://github.com/facebookresearch/codellama
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stip_original import MultiHeadAttention
import copy

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

class LlamaTransformerBlock(nn.Module):
    """
    """
    def __init__(self, d_model, num_heads, d_ff):
        super(LlamaTransformerBlock, self).__init__()
        self.attention_layer = MultiHeadAttention(d_model, num_heads)
        self.ff_layer = FeedForward(d_model, d_ff)
        self.rmsn1 = RMSNorm(d_model)
        self.rmsn2 = RMSNorm(d_model)

    def forward(self, x, mask):
        attention_input = self.rmsn1(x)
        attention_output = self.attention_layer(attention_input, attention_input, attention_input, mask)
        h = x + attention_output

        ff_input = self.rmsn2(h)
        ff_output = self.ff_layer(ff_input)
        res = h + ff_output
        return res
    

def permute_block(blk, p):
    p_blk = copy.deepcopy(blk)
    with torch.no_grad():
        for name, para in p_blk.named_parameters():
            # print(name)
            if name in ["attention_layer.w_q.weight",
                        "attention_layer.w_k.weight",
                        "attention_layer.w_v.weight",
                        "ff_layer.w1.weight",
                        "ff_layer.w3.weight"]:
                para.data = para.data[:, p]
            if name in ["attention_layer.w_o.weight",
                        "attention_layer.w_o.bias",
                        "ff_layer.w2.weight"]:
                para.data = para.data[p]
    return p_blk


if __name__ == "__main__":
    BS = 2
    SEQLEN = 3
    DMODEL = 4
    DFF = 8
    NHEADS = 2

    TEST_BLK = 1
    
    if TEST_BLK:
        x = torch.from_numpy(np.random.rand(BS, SEQLEN, DMODEL)).float()
        block = LlamaTransformerBlock(DMODEL, NHEADS, DFF)
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
