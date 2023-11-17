# Secure Transformer Inference

Secure Transformer Inference Protocol, STIP, is a three-party protocol that can protect both Transformer parameters and user data during the inference phase. 
For each feedforward inference process, STIP only introduces permutation computation of input and output data on the user side. 
STIP can be applied to real-world services like ChatGPT.

The figure below shows the overview of STIP.

<img src="./.github/stip_overview.png" alt="overview" width="500"/>

We consider three parties:

* Party-1 ($P_1$): Model developer (e.g., OpenAI) that owns the original Transformer model $f_\theta$.
* Party-2 ($P_2$): Cloud computing platform (e.g., Azure) that owns the computing hardware.
* Party-3 ($P_3$): Users that own private input (e.g., prompt token embedding) and output (e.g., response token logits).

Initialization phase:
* $P_1$ randomly generate $\pi \in \mathbb{R}^{d\times d}$
* $P_1$ transform $f_\theta$ to $f_{\theta'}$ using $\pi$
* $P_1$ send $f_{\theta'}$ to $P_2$ and send $\pi$ to $P_3$

Inference phase:
* $P_3$ transform $x$ to $x'=x\pi$ and send $x'$ to $P_2$
* $P_2$ compute $f_{\theta'}(x')=y'$ and send $y'$ to $P_3$
* $P_3$ de-transform $y'$ by computing $y'\pi^T$ and get $y\pi\pi^T=y$

For detailed transformation of model parameters, please refer to [our paper](./preprint_Secure_Transformer_Inference.pdf). 

## Test Code

We tested original Transformer ([Vaswani, Ashish, et al. 2017](https://arxiv.org/abs/1706.03762)) and Llama Transformer ([Touvron, Hugo, et al. 2023](https://arxiv.org/abs/2302.13971)) using PyTorch.

The test logic is simple: transform the model and re-transform the inference result, then check the absolute difference (Considering the representation error of floating point numbers, not checking for equality) between it and the original result.

## Citation

If you find STIP helpful, please consider citing:
```
@misc{cryptoeprint:2023/1763,
      author = {Mu Yuan and Lan Zhang and Xiang-Yang Li},
      title = {Secure Transformer Inference},
      howpublished = {Cryptology ePrint Archive, Paper 2023/1763},
      year = {2023},
      note = {\url{https://eprint.iacr.org/2023/1763}},
      url = {https://eprint.iacr.org/2023/1763}
}
```

## License

Secure Transformer Inference Protocol (STIP) is licensed under the [MIT License](./LICENSE).
