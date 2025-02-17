# Typological_Universal

Verify the innate preferences of typological universals in language models using counterfactual English and Japanese grammars.

## Dataset

We used English and Japanese splits in Wiki-40b.

Local: Download the data into text files by running `src/data_processing/wiki_40b.py`

Slurm-based cluster: Continue reading the following sections.

## Parser

We used Stanza to obtain dependency parses.

## Preprocessing to get clean data

Run the following command before you start training on a dataset, if you haven't downloaded Wiki-40b before.
Note that we need to create two different virtual environments to avoid package conflicts between Tensorflow and PyTorch.

```
module load eth_proxy gcc/8.2.0 python_gpu/3.9.9
python -m venv env1
source ./env1/bin/activate

pip install --upgrade pip
pip install --ignore-installed --no-cache-dir -r ./src/data_processing/requirements.txt

./scripts/data.sh -<language_code>
```

If permission denied, try executing:

```
chmod -R +x ./scripts
chmod -R +x ./src
```

For the Greenberg word-order correlation universals, we experimented on Japanese (SOV) and English (SVO).

## Creating environment on Slurm

The experiments are conducted on ETH Cluster (Euler). 

The commands should fit to every Slurm-based HPC cluster with some slight modifications cluster-wise.

Make sure you are in the root of your project.

If you have activated a virtual environment already, run the following command:

```
deactivate
```

If you want to delete a virtual environment and the packages it contains, run the following command:

```
rm -r <env_folder_name>
```

Then:

```
module load eth_proxy gcc/8.2.0 python_gpu/3.9.9
python -m venv env2
source ./env2/bin/activate

pip install --upgrade pip
```

Whenever you install a new package, make sure the correct venv is activated!

To install new packages:

```
pip install --no-cache-dir <package_names>
```

If you need a different version of an existing package, then you can also install it in your virtual environment. For instance for installing a newer numpy version:

```
OPENBLAS=$OPENBLAS_ROOT/lib/libopenblas.so pip install --ignore-installed --no-deps numpy==1.20.0
```

To install required packages for this project:

```
OPENBLAS=$OPENBLAS_ROOT/lib/libopenblas.so pip install --ignore-installed --no-cache-dir -r requirements.txt
```

You might also want to login to your wandb account (only once): ```wandb login```

## Training

Before running the scripts, make sure you modify the *.euler files accordingly:

* Modify SBATCH options (e.g. resource requests) 
* Modify the venv source path: ```source <path_to_project>/env2/bin/activate```

Run the scripts from the project root.

```
### See help for all the available options ###
./scripts/train.sh -h

### Train a model with default configuration ###
./scripts/train.sh

### Train a model with custom configuration ###
./scripts/train.sh -n <model_name> -d <dataset> -l <lang> -s <seed> -p <project_name> -t <tokenizer_path> -c <ckpt_path> -f <configuration_file> -T <test_mode> -w <sweep_id>
```
