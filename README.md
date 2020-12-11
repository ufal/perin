<h1 align="center"><b>PERIN: Permutation-invariant Semantic Parsing</b></h1>

<p align="center">
  <i><b>David Samuel & Milan Straka</b></i>
</p>

<p align="center">
  <i>
    Charles University<br>
    Faculty of Mathematics and Physics<br>
    Institute of Formal and Applied Linguistics
  </i>
</p>
<br>

<p align="center">
  <a href="https://arxiv.org/abs/2011.00758"><b>Paper</b></a><br>
  <a href="https://drive.google.com/drive/folders/11ozu_uo9z3wJwKl1Ei2C3aBNUvb66E-2?usp=sharing"><b>Pretrained models</b></a><br>
  <a href="https://colab.research.google.com/drive/1jqYATZgv1GhEto4eUKH7_-P8VmHIKGMs?usp=sharing"><b>Interactive demo on Google Colab</b></a>
</p>

<p align="center">
  <img src="img/illustration.png" alt="Overall architecture" width="340"/>  
</p>

_______

<br>

PERIN is a universal sentence-to-graph neural network architecture modeling semantic representation from input sequences.

The main characteristics of our approach are:

- <b>Permutation-invariant model</b>: PERIN is, to our best
  knowledge, the first graph-based semantic parser that predicts all nodes at once in parallel and trains them with a permutation-invariant loss function.
- <b>Relative encoding</b>: We present a substantial improvement of relative encoding of node labels, which allows the use of a richer set of encoding rules.
- <b>Universal architecture</b>: Our work presents a general sentence-to-graph pipeline adaptable for specific frameworks only by adjusting pre-processing and post-processing steps.


Our model was ranked among the two winning systems in both the *cross-framework* and the *cross-lingual* tracks of [MRP 2020](http://mrp.nlpl.eu/2020/) and significantly advanced the accuracy of semantic parsing from the last year's MRP 2019. 

_______

<br>

This repository provides the official PyTorch implementation of our paper "[ÚFAL at MRP 2020: Permutation-invariant Semantic Parsing in PERIN]()" together with [pretrained *base* models](https://drive.google.com/drive/folders/11ozu_uo9z3wJwKl1Ei2C3aBNUvb66E-2?usp=sharing) for all five frameworks from [MRP 2020](http://mrp.nlpl.eu/2020/): AMR, DRG, EDS, PTG and UCCA.

_______

<br>

## How to run

### :feet: &nbsp; Clone repository and install the Python requirements

```sh
git clone https://github.com/ufal/perin.git
cd perin

pip3 install -r requirements.txt 
pip3 install git+https://github.com/cfmrp/mtool.git#egg=mtool
```

### :feet: &nbsp; Download and pre-process the dataset

Download the [treebanks](http://mrp.nlpl.eu/2020/index.php?page=14) into `${data_dir}` and split the cross-lingual datasets into training and validation parts by running:
```sh
./scripts/split_dataset.sh "path_to_a_dataset.mrp"
```

Preprocess and cache the dataset (computing the relative encodings can take up to several hours):
```sh
python3 preprocess.py --config config/base_amr.yaml --data_directory ${data_dir}
```

You should also download [CzEngVallex](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1512) if you are going to parse PTG:
```sh
curl -O https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1512/czengvallex.zip
unzip czengvallex.zip
rm frames_pairs.xml czengvallex.zip
```

### :feet: &nbsp; Train

To train a shared model for the English and Chinese AMR, run the following script. Other configurations are located in the `config` folder.
```sh
python3 train.py --config config/base_amr.yaml --data_directory ${data_dir} --save_checkpoints --log_wandb
```

### :feet: &nbsp; Inference

You can run the inference on the validation and test datasets by running:
```sh
python3 inference.py --checkpoint "path_to_pretrained_model.h5" --data_directory ${data_dir}
```

## Citation

```
@inproceedings{Sam:Str:20,
  author = {Samuel, David and Straka, Milan},
  title = {{{\'U}FAL} at {MRP}~2020:
           {P}ermutation-Invariant Semantic Parsing in {PERIN}},
  booktitle = CONLL:20:U,
  address = L:CONLL:20,
  pages = {\pages{--}{53}{64}},
  year = 2020
}
```
