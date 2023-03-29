# NAS-Benchmark-Evaluation

## Usage 

Create a conda environment using the env.yml file

```bash
conda env create -f env.yml
```

Activate the environment and follow the instructions to install
```
conda activate bencheval
```

Download:
- NASbench101 data (see https://github.com/google-research/nasbench)
- NASbench201 data (see https://github.com/D-X-Y/NAS-Bench-201) 
- TransNAS-Bench-101 (see https://github.com/yawen-d/TransNASBench)
and place them under the datasets folder


To process the benchmark search space and generate pickle files with information about all architectures:
```
python diversity*.py
```

With the pickle files, the file plots_diversity*.py allows creating all the plots and tables shown in the paper.


The code is licensed under the MIT licence.


## Acknowledgements

This repository makes liberal use of code from the [AutoDL](https://github.com/D-X-Y/AutoDL-Projects) library, [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201), [[NAS-WOT](https://github.com/BayesWatch/nas-without-training). We are grateful to the authors for making the implementations publicly available.
