# Character Queries: A Transformer-based Approach to On-Line Handwritten Character Segmentation

[![arXiv badge][arxiv-badge]][arxiv]
[![DOI badge][doi-badge]][doi]
[![Python][actions-python-badge]][actions-python-link]

![Model Overview](https://user-images.githubusercontent.com/3986846/235125728-2446a968-25c6-49f2-9f26-4ae0da4a02f2.png)

## Table of Contents

<!-- vim-markdown-toc GitLab -->

* [Models](#models)
* [Ground Truth](#ground-truth)
  * [Convert to JSON](#convert-to-json)
* [Requirements](#requirements)
  * [Manually with pip](#manually-with-pip)
* [Usage](#usage)
  * [Training](#training)
    * [Logs](#logs)
  * [Exporting Model](#exporting-model)
* [Development](#development)
  * [Pre-Commit Hooks](#pre-commit-hooks)
  * [Debugger](#debugger)
* [License](#license)
* [Citation](#citation)

<!-- vim-markdown-toc -->

## Models

Download the model checkpoints:

| Model                                                | #params | IAM Test Set F | IAM Test Set F | VNOnDB Test Set |
|------------------------------------------------------|---------|----------------|----------------|-----------------|
| [Character Query Transformer][model-character-query] | 6.47M   | 92.28          | 95.11          | 92.06           |


## Ground Truth

The ground truth segmentation annotations for [IAM-OnDB][iam-on] and [HANDS-VNOnDB][vnondb] can be
[downloaded from SWITCHDrive][gt] or with the direct links to each file in the table below.

*Note: This does not contain the on-line handwriting dataset themselves, but only the ground truth segmentation
annotations. The IAM-OnDB and HANDS-VNOnDB datasets need to be downloaded separately.*

| Dataset                | File needed                         | GT Train                                             | GT Validation                                                  | GT Test                                                                              |
|------------------------|-------------------------------------|------------------------------------------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------|
| [IAM-OnDB][iam-on]     | [lineStrokes-all.tar.gz][iam-lines] | [trainset_segmented.json][iam-gt-train]              | [testset_v_segmented.json][iam-gt-validation]                  | [testset_t_segmented.json][iam-gt-test-t], [testset_f_segmented.json][iam-gt-test-f] |
| [HANDS-VNOnDB][vnondb] | [InkData_word.zip][vnondb-word]     | [InkData_word_train_segmented.json][vnondb-gt-train] | [InkData_word_validation_segmented.json][vnondb-gt-validation] | [InkData_word_test_segmented.json][vnondb-gt-test]                                   |

*Direct downloads when clicking on the zip/tar.gz file requires you to be logged in on their website, it might be
necessary to click on the dataset link an log in from there.*


### Convert to JSON

In order to use the ground truth, it needs to be combined with the corresponding data points from the respective
datasets. The data points are stored in a single archive (zip/tar.gz) in InkML files, to make it simpler to use for the
data loading, they are converted to individual JSON files, which contain both the segmentation annotations as well as
all necessary point information. It can be converted with the `convert_gt.py` script as follows:

```sh
# IAM-OnDB (train, validation, test set T and test set F)
python convert_gt.py -d data/iam/lineStrokes-all.tar.gz -s data/gt/iam/trainset_segmented.json path/to/gt/iam/testset_v_segmented.json data/gt/iam/testset_t_segmented.json data/gt/iam/testset_f_segmented.json -o data/converted/iam

# HANDS-VNOnDB (train, validation and test set)
python convert_gt.py -d data/vnondb/InkData_word.zip -s data/gt/vnondb/InkData_word_train_segmented.json data/gt/vnondb/InkData_word_validation_segmented.json data/gt/vnondb/InkData_word_test_segmented.json -o data/converted/vnondb -t vnondb
```

After this there is one directory for each subset and a corresponding `.tsv` file, which can be used as an index. The
following file structure is produced with the aforementioned commands:

```
data/converted/
├── iam
│   ├── testset_f_segmented/
│   │   ├── a01-013z-01.xml.json
│   │   ...
│   ├── testset_f_segmented.tsv
│   ├── testset_t_segmented/
│   │   ├── a01-000u-03.xml.json
│   │   ...
│   ├── testset_t_segmented.tsv
│   ├── testset_v_segmented/
│   │   ├── a01-003-01.xml.json
│   │   ...
│   ├── testset_v_segmented.tsv
│   ├── trainset_segmented/
│   │   ├── a01-001w-02.xml.json
│   │   ...
│   └── trainset_segmented.tsv
└── vnondb
    ├── InkData_word_test_segmented/
    │   ├── 20151208_0146_7105_1.inkml_0.json
    │   ...
    ├── InkData_word_test_segmented.tsv
    ├── InkData_word_train_segmented/
    │   ├── 20140603_0003_BCCTC.inkml_0.json
    │   ...
    ├── InkData_word_train_segmented.tsv
    ├── InkData_word_validation_segmented/
    │   ├── 20151224_0141_7818_1.inkml_0.json
    │   ...
    └── InkData_word_validation_segmented.tsv

```

## Requirements

The dependencies can automatically be installed with the `install_requirements.py` script.

```sh
python install_requirements.py
```

It installs all dependencies listed in `requirements.txt` and dev dependencies (checker, linter, formatter).

Optionally, the targets to install can be specified as arguments:

```sh
# Install all dependencies
python install_requirements.py deps dev

# Equivalent to the above (default)
python install_requirements.py all
```

For convenience: `all = [deps, dev]`.

### Manually with pip

All dependencies can be installed manually with pip.

```sh
pip install -r requirements.txt
```

On *Windows* the PyTorch packages may not be available on PyPi, hence you need to point to the official PyTorch
registry:

```sh
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

If you'd like to use a different installation method or another CUDA version with PyTorch follow the instructions on
[PyTorch - Getting Started][pytorch-started].

## Usage

### Training

Training is done with the `train.py` script:

```sh
python train.py --name some-name --train-gt /path/to/gt.tsv --validation-gt /path/to/gt.tsv difficult=/another/path/some-gt.tsv --chars /path/to/chars.tsv --fp16 --ema
```

The `--name` option is used to give it a name, otherwise the current date and time is used as a name. Bes
resume from the given checkpoint, if not specified it starts fresh.

Multiple validation datasets can be specified, optionally with a name,  `--validation-gt /path/to/gt.tsv
difficult=/another/path/some-gt.tsv` would use two validation sets. When no name is specified, the name of the ground
truth file and its parent directory is used. In the previous example the two sets would have the names: `to/gt` and
`difficult`.
The best checkpoints are determined by the average across all validation sets.

In order to know which characters are available to the model, the `--chars` option needs to be the path to a TSV file
with a list of characters, where each character is on a new line. It is a TSV file because that allows to have multiple
columns, where the first column is the character and any additional column will be ignored, which can be useful when
storing statistics of the occurring characters within the dataset in the additional columns. A simple text file with
just the characters, each on its own line, works just as well.

Modern GPUs contain Tensor Cores (starting from V100 and RTX series) which enable mixed precision calculation, using
optimised fp16 operations while still keeping the fp32 weights and therefore precision.

It can be enabled by setting the `--fp16` flag.

*Other GPUs without Tensor Cores do not benefit from using mixed precision since they only do fp32 operations and you
may find it even becoming slower.*

The `--ema` flag enables the Exponential Moving Average (EMA) of the model parameters, which helps stabilise the final
model and is recommend to always use.



For all options see `python train.py --help`.

**Training a Character Query Transformer**

Most of the default values are set for the character queries, so only a few additional arguments are needed besides
specifying the datasets.

```sh
python train.py \
    --name character-queries-iam-and-vnondb \
    --gt-train data/converted/combined-iam-vnondb/train.tsv \
    --gt-validation \
        IAM_Validation=data/converted/iam/testset_v_segmented.tsv \
        VNONDB_Validation=data/converted/vnondb/InkData_word_validation_segmented.tsv \
    --chars data/converted/combined-iam-vnondb/chars.tsv \
    -b $BATCH_SIZE \
    --fp16 \
    --ema \
    --features-normalise
```

**Training an LSTM**

For the LSTM more options need to be changed.

```sh
python train.py \
    --name lstm-iam-and-vnondb \
    --gt-train data/converted/combined-iam-vnondb/train.tsv \
    --gt-validation \
        IAM_Validation=data/converted/iam/testset_v_segmented.tsv \
        VNONDB_Validation=data/converted/vnondb/InkData_word_validation_segmented.tsv \
    --chars data/converted/combined-iam-vnondb/chars.tsv \
    -b $BATCH_SIZE \
    --fp16 \
    --ema \
    --features x:delta y:delta index stroke global_index ctc_spike:embed \
    -l 3e-3 \
    --lr-warmup 5 \
    --activation relu \
    -m rnn
```

#### Logs

During the training various types of logs are created with [Lavd][lavd] and everything can be found in `log/` and is
grouped by the experiment name.

- Summary
- Checkpoints
- Top 5 Checkpoints
- Event logs

To visualise the logged data run:

```sh
lavd log/
```

### Exporting Model

A model can be exported (JIT compiled) such that it can be used in Python or C++ directly without having to manually
define the models. It can be loaded directly with [`torch.jit.load`][pytorch-jit-load] in Python or with the equivalent
function `torch::jit::load` in C++.


```sh
python export_model.py -c log/some-name/best/ -o exported/best-model.ptc
```

When a directory is given to `-c`/`--checkpoint` instead of the model checkpoint directly, it will automatically look
for the `model.pt` file in that directory.

The exported model will be saved where to the path given to `-o`/`--output` or if not specified, the model will be saved
as `exported/{model-kind}-{date}.ptc`, e.g. `exported/rnn-2022-03-22.ptc`.
It is recommended to use the file extension `.ptc`, where the `c` stands for *compiled*, in order to easily distinguish
the exported models from saved checkpoints.

## Development

To ensure consistency in the code, the following tools are used and also verified in CI:

- `ruff`: Linting
- `mypy`: Type checking
- `black`: Formatting
- `isort`: Import sorting / formatting

All of these tools are installed with the installation script, when all dependencies are installed and are also
available with the `dev` group:

```sh
python install_requirements.py

# Or only the these dev tools
python install_requirements.py dev
```

It is recommended to have an editor configured such that it uses these tools, for example with the Python language
server, which uses the [Language Server Protocol (LSP)][lsp], which allows you to easily see the errors / warnings and
also format the code (potentially, automatically on save) and other helpful features.

Almost all configurations are kept at their default, but because of conflicts, a handful of them needed to be changed.
These modified options are configured in `pyproject.toml`, hence if your editor does not agree with CI, it is most likely due
to the config not being respected, or by using a different tool that may be used as a substitute.

### Pre-Commit Hooks

All checks can be run on each commit with the Python package `pre-commit`.

First it needs to be installed:

```sh
pip install pre-commit
```

And afterwards the git pre-commit hooks need to be created:

```sh
pre-commit install
```

From now on, the hook will run the checks automatically for the changes in the commit (not all files).

However, you can run the checks manually on all files if needed with the `-a`/`--all` flag:

```sh
pre-commit run --all
```

### Debugger

Python's included debugger `pdb` does not work for multi-processing and just crashes when the breakpoint is reached.
There is a workaround to make it work with multiple processes, which is included here, but it is far from pleasant to
use since the same TTY is shared and often alternates, making the debugging session frustrating, especially since the
readline features do not work with this workaround.

A much better debugger uses the [Debugger Adapter Protocol (DAP)][dap] for remote debugging, which allows to have a full
debugging experience from any editor that supports DAP. In order to enable this debugger you need to have `debugpy`
installed.

```sh
pip install debugpy
```

To start a debugging sessions, a breakpoint needs to be set with custom breakpoint function defined in `debugger.py`:

```py
from debugger import breakpoint

# ...
breakpoint("Optional Message")
```

This will automatically enable the debugger at the specified port (default: 5678) and for every additional process, it
will simply create a different session, with the port incremented by one.

*If `debugpy` is not installed, it will fall back to the multi-processing version of PDB.*.

Should your editor not support DAP (e.g. PyCharm doesn't and probably won't ever), it is easiest to use VSCode for this.

## License

The code and models are released under the [MIT License](LICENSE.md) and the ground truth annotation data is licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@inproceedings{jungo2023characterqueries,
  author={Jungo, Michael and Wolf, Beat and Maksai, Andrii and Musat, Claudiu and Fischer, Andreas},
  title={Character Queries: A Transformer-Based Approach to On-line Handwritten Character Segmentation},
  doi={10.1007/978-3-031-41676-7_6},
  booktitle={Document Analysis and Recognition - ICDAR 2023},
  editor={Fink, Gernot A. and Jain, Rajiv and Kise, Koichi and Zanibbi, Richard},
  year={2023},
  publisher={Springer Nature Switzerland},
  pages={98--114},
  isbn={978-3-031-41676-7}
}
```

[actions-python-badge]: https://github.com/jungomi/character-queries/actions/workflows/python.yml/badge.svg
[actions-python-link]: https://github.com/jungomi/character-queries/actions/workflows/python.yml
[arxiv]: https://arxiv.org/abs/2309.03072
[arxiv-badge]: https://img.shields.io/badge/arXiv-2309.03072-b31b1b.svg
[dap]: https://microsoft.github.io/debug-adapter-protocol/
[doi]: https://doi.org/10.1007/978-3-031-41676-7_6
[doi-badge]: https://zenodo.org/badge/DOI/10.1007/978-3-031-41676-7_6.svg
[gt]: https://drive.switch.ch/index.php/s/qAfBtb1P05heJWZ
[iam-gt-test-f]: https://drive.switch.ch/index.php/s/4fx1EY4fqEdJbss/download
[iam-gt-test-t]: https://drive.switch.ch/index.php/s/eDOyjldldL17ev9/download
[iam-gt-train]: https://drive.switch.ch/index.php/s/SSxan3pm1WHdQZk/download
[iam-gt-validation]: https://drive.switch.ch/index.php/s/XgXfWVSFrv6oLwz/download
[iam-lines]: https://fki.tic.heia-fr.ch/DBs/iamOnDB/data/lineStrokes-all.tar.gz
[iam-on]: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database
[lavd]: https://github.com/jungomi/lavd
[lsp]: https://microsoft.github.io/language-server-protocol/
[model-character-query]: https://drive.switch.ch/index.php/s/NMYy89LcfEbJIK8/download
[pytorch]: https://pytorch.org/
[pytorch-started]: https://pytorch.org/get-started/locally/
[pytorch-jit-load]: https://pytorch.org/docs/stable/generated/torch.jit.load.html
[vnondb]: https://tc11.cvc.uab.es/datasets/HANDS-VNOnDB2018_1/
[vnondb-gt-test]: https://drive.switch.ch/index.php/s/7bEBkfQh7ftPgJF/download
[vnondb-gt-train]: https://drive.switch.ch/index.php/s/NhflPmbr5NQ4F8C/download
[vnondb-gt-validation]: https://drive.switch.ch/index.php/s/BLa6mVjYM3vy71M/download
[vnondb-word]: https://tc11.cvc.uab.es/index.php?com=upload&action=file_down&section=dataset&section_id=195&file=182
