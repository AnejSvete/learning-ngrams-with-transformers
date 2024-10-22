# Can Transformer Language Models Learn $n$-gram Language Models?

## Getting started with the code
Clone the repository:
```bash
$ git clone https://github.com/rycolab/learning-ngrams
$ cd learning-ngrams
```
It may be beneficial to create a new Python virtual environment (e.g., using conda). 
We aim at Python 3.10 version and above.

Then install the package (in the required package) in editable mode:
```bash
$ pip install -e .
```
We use pytest to unit test the code.


## Generating the data

1. Set the configuration file `dataset_generation.yaml` in the `config` directory to specify the parameters for the dataset generation including the output directory.
2. Run the following command from the base directory of the repository to generate the dataset:
```bash
$ bash ./scripts/generate_datasets.sh    
```
