<h1>Ghost Demo</h1>

<h3>Install the below required packages and download the required data</h3>

```bash
conda create -n LMGIA python=3.12.3
conda activate LMGIA
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 datasets==2.19.1 evaluate==0.4.2 accelerate==0.30.1 nltk==3.8.1 spacy==3.8.2 absl-py==2.1.0 rouge_score==0.1.2 scikit-learn==1.6.0 bitsandbytes==0.45.0 peft==0.14.0
python -m spacy download en_core_web_sm
python
```

```python
import nltk
nltk.download('punkt')
```

<h3>In the transformation folder, create a symbolic link to the utils folder</h3>

```bash
cd transformation
ln -s ../utils utils
```

<h3>Run the below command to see a Ghost demo</h3>

```bash
python transform_data_discrete_bert_full.py
```