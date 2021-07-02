# Huawei Cup Final 2020 AI nomination

## First place solution

Here is my code for Huawei Cup 2020 competition, that gained the first prize.

The task was to generate low-resolution images from high-resolution ones so that the model trained on the resulting dataset for super-resolution task has the highest quality.
The dataset consisted of low-resolution and high-resolution images.

The solution is based on RRDB architecture (https://arxiv.org/pdf/1809.00219.pdf) of CNN with DISTS (https://arxiv.org/pdf/2004.07728.pdf) loss-function.

Note: the experiments were carried out on a "clean" dataset, in which unpaired images were discarded (there were such images in the original dataset!).
