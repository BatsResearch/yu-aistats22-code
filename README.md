# NPLM Experiments
NPLM is a programmatic weak supervision system that supports (partial) labeling functions with supervision granularites ranging from class to a set of classes.

For the underlying weak supervision system, further documentation and tutorials, please see [NPLM repo](https://github.com/BatsResearch/nplm).

To use the experimental code, please first install the nplm package from [NPLM repo](https://github.com/BatsResearch/nplm).

This is the codebase for the experiments mentioned in [Learning from Multiple Noisy Partial Labelers](https://arxiv.org/pdf/2106.04530.pdf).

## Code Organization & General Instructions

The code is organized by task, please refer to corresponding <dataset_name>_pipeline.ipynb and run_end_model_<dataset_name>.py (for text tasks). Please refer to votes for plf votes for the image tasks and annotators/ for interfaces and plfs for bot image and text tasks.
For Object detection (vision) tasks, label modeling and end model training are included in <dataset_name>_pipeline.ipynb.
For text classification tasks, PLFs curation and label modeling are in <dataset_name>_pipeline.ipynb, and it can produce the probabilistic labels to train the end model with run_end_model_<dataset_name>.py.

## Environment Setupinstall
```
conda create --name nplm python=3.7
conda activate nplm
pip install -r requirements.txt
git clone https://github.com/BatsResearch/nplm.git
cd nplm; pip install .
```

## Data
Preprocessed text data is included in this repo. To download the image data, please run 
```
./download_image_dataset.sh
```

## Checklist
- [x] Code (v0.1)
- [x] Text Data
- [ ] Image Data (Configuring Data Hosting)
- [ ] Instructions
- [ ] Documentation


## Citation

Please cite the following paper if you are using our tool. Thank you!

[Peilin Yu](http://www.yupeilin.com), [Tiffany Ding](https://tiffanyding.github.io/), [Stephen H. Bach](http://cs.brown.edu/people/sbach/). "Learning from Multiple Noisy Partial Labelers". Artificial Intelligence and Statistics (AISTATS), 2022.

```
@inproceedings{yu2022nplm,
  title = {Learning from Multiple Noisy Partial Labelers}, 
  author = {Yu, Peilin and Ding, Tiffany and Bach, Stephen H.}, 
  booktitle = {Artificial Intelligence and Statistics (AISTATS)}, 
  year = 2022, 
}
```
