## Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time

Large language models(LLMs) have sparked a new wave of exciting AI applications. Hosting these models at scale requires significant memory resources. One crucial memory bottleneck for the deployment stems from the context window. It is commonly recognized that model weights are memory hungry; however, the size of key-value embedding stored during the generation process (KV cache) can easily surpass the model size. The enormous size of the KV cache puts constraints on the inference batch size, which is crucial for high throughput inference workload. Inspired by an interesting observation of the attention scores, we hypothesize the persistence of importance: only pivotal tokens, which had a substantial influence at one step, will significantly influence future generations. Based on our empirical verification and theoretical analysis around this hypothesis, we propose Scissorhands, a system that maintains the memory usage of the KV cache at a fixed budget without finetuning the model. In essence, Scissorhands manages the KV cache by storing the pivotal tokens with a higher probability. We validate that Scissorhands reduces the inference memory usage of the KV cache by up to 5X without compromising model quality. We further demonstrate that Scissorhands can be combined with 4-bit quantization, traditionally used to compress model weights, to achieve up to 20X compression.

[Paper Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/a452a7c6c463e4ae8fbdc614c6e983e6-Paper-Conference.pdf)

## Accuracy Benchmark
We based our accuracy benchmark based on Decentralized_FM_alpha([https://github.com/DS3Lab/Decentralized_FM_alpha](https://github.com/DS3Lab/Decentralized_FM_alpha))

**Requirements**

```
    pip3 install --pre torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    pip3 install cupy-cuda11x==11.0.0
    python3 -m cupyx.tools.install_library --cuda 11.x --library nccl
    pip3 install transformers
```
**Perplexity on c4**

We provide script to prepare data in `Scissorhands/Decentralized_FM_alpha/c4_val/getdata.py`

To run evaluation without compressed KV cache
```Scissorhands/Decentralized_FM_alpha/run_infer_opt_66b_c4.sh```

To run evaluation with compressed KV cache
```Scissorhands/Decentralized_FM_alpha/run_infer_opt_66b_sparse_c4.sh```

You will need to specify 
(1) the model checkpoint path
(2) c4 data path

**Coming Soon**
 - [ ] Fewshot evaluation Code
 - [ ] Generation Code
