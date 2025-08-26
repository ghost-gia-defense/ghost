# GHOST Demo

This is a demo for the GHOST defense against gradient inversion attacks for language models. This demo is on BERT model and DAGER attack. Full supports for other models and attacks will be released upon publication.
The code is tested on Linux system with one NVIDIA H100 GPU.

## Install
First, make sure you have conda installed. You can download and
install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) corresponding to your system (Windows, Linux, Mac) from their
official websites.

Then, in terminal on Mac or Linux, or in Anaconda Prompt in Windows, run the following commands to create a new conda environment and activate it.

```bash
conda create -n GHOST python=3.12.3
conda activate GHOST
```

You need to install torch. Our expected version is `2.3.0`. You need to check
the [official PyTorch website](https://pytorch.org/get-started/previous-versions/) to install the corresponding one to your system (Windows, Linux, Mac).

Then, run the following commands to install the required packages.
```bash
pip install transformers==4.44.2 datasets==2.19.1 evaluate==0.4.2 accelerate==0.30.1 nltk==3.8.1 spacy==3.8.2 absl-py==2.1.0 rouge_score==0.1.2 scikit-learn==1.6.0 bitsandbytes==0.45.0 peft==0.14.0
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

<h3>Run the below command to see a GHOST obfuscation demo</h3>

```bash
cd obfuscation
python obfuscate_bert.py
```