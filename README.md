
## RTuckER: Riemannian Optimization for Link Prediction Using Tensor Decompositions

This repository contains PyTorch implementation of RTuckER model for knowledge graph link prediction task. 

### Summary

The proposed method is a modification of the approach described in paper TuckER[1]. It represents the knowledge graph as a tensor with a fixed multilinear rank and uses the Riemannian optimization approach for training.
Unlike TuckER model, this approach doesn't employ such common DL techniques as Dropout or BatchNormalization.

### Repository structure

There are two types of models:

* `symmetric` -- model uses equal subjects and objects embeddings in Tucker decomposition.
*  `asymmetric` -- otherwise.

### Parameters of setup

Dataset | rank | lr | reg | batch size | decay rate | symmetric
:--- | :---: | :---: | :---: | :---: | :---:  | :---:
WN18RR | (200, 20, 200) | 0.1 | 1e-10 | 2048 | 0.999 |True
FB15k-237 | (200, 200, 200) | 0.1 | 1e-10 | 512|0.999 |True


### Link prediction results

Dataset | MRR | Hits@10 | Hits@3 | Hits@1
:--- | :---: | :---: | :---: | :---:
WN18RR | 0.449 | 0.508 | 0.468 | 0.414
FB15k-237 | 0.313 | 0.481 | 0.342 | 0.231

### References

[TuckER: Tensor Factorization for Knowledge Graph Completion](https://arxiv.org/pdf/1901.09590.pdf)  

### License

MIT License
