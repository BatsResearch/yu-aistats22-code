# NPLM Experiments
NPLM is a programmatic weak supervision system that supports (partial) labeling functions with supervision granuarity ranging from class to a set of classes.

For the underlying weak supervision system, further documentation and tutorials, please see [NPLM repo](https://github.com/BatsResearch/nplm).

This is the codebase for the experiments mentioned in [Learning from Multiple Noisy Partial Labelers](https://arxiv.org/pdf/2106.04530.pdf).

## Code Organization

The code is organized by task, please refer to corresponding <datset_name>_pipeline.ipynb and run_end_model_<dataset_name>.py (for text tasks). Please refer to votes/ for plf votes for the image tasks and annotators/ for interfaces and plfs for bot image and text tasks.

## Setup
```
conda create --name nplm python=3.7
conda activate nplm
pip install -r requirements.txt
```

## Data

## Checklist
- [] Code
- [] Data
- [] Documentation


## Citation

Please cite the following paper if you are using our tool. Thank you!

[Peilin Yu](https://www.yupeilin.com), [Tiffany Ding](https://tiffanyding.github.io/), [Stephen H. Bach](http://cs.brown.edu/people/sbach/). "Learning from Multiple Noisy Partial Labelers". Artificial Intelligence and Statistics (AISTATS), 2022.

```
@inproceedings{yu2022nplm,
  title = {Learning from Multiple Noisy Partial Labelers}, 
  author = {Yu, Peilin and Ding, Tiffany and Bach, Stephen H.}, 
  booktitle = {Artificial Intelligence and Statistics (AISTATS)}, 
  year = 2022, 
}
```
