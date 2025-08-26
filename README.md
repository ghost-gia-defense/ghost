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

For Linux, run the following command to install torch.

```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

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
```
usage: Obfuscate data with GHOST [-h] [--dataset DATASET] [--device DEVICE] [--num_of_samples NUM_OF_SAMPLES] [--model_name MODEL_NAME] [--beam_width BEAM_WIDTH] [--overlap OVERLAP] [--topk TOPK] [--recover_batch RECOVER_BATCH]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     The target dataset
  --device DEVICE       The device to use
  --num_of_samples NUM_OF_SAMPLES
                        The number of samples being obfuscated
  --model_name MODEL_NAME
                        The model being fine-tuned
  --beam_width BEAM_WIDTH
                        The beam width to use in GHOST
  --overlap OVERLAP     The overlap ratio between neighbours to use in GHOST
  --topk TOPK           The top-k neighbours being evaluated in GHOST
  --recover_batch RECOVER_BATCH
                        The batch to recover from in case of interruption
```

By default, this script will obfuscate `10` samples from the `SST-2` dataset with the `bert-base-uncased` model. This is for demonstration purposes so that you can quickly see how GHOST obfuscates data. You can change the `--num_of_samples` argument to obfuscate more samples. The obfuscated data will be saved in the `data/bert-base-uncased` folder in json format.