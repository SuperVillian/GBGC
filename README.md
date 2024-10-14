# GBGC
This is the official implementation for "GBGC: Efficient and Adaptive Graph Coarsening viaGranular-Ball Computing" (NIPS 2024).

## dataset
because "OVCAR-8H","P388H","SF-295H" are very large.
You can get the dataset from https://chrsmrrs.github.io/datasets/docs/datasets/
And put the dataset in dataset folder

## Run Code

The initialization directory is the root directory `./`.

```
experiment 1 acc and time
nohup python -u GB_coarsening.py > ./result/resultGBGC.log 2>&1 &

experiment 2 different Coarsening rate
nohup python -u GBGC_ratio.py > ./result/GBGC_ratio.log 2>&1 &