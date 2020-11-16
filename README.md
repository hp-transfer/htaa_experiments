# Hyperparameter Transfer Across Developer Adjustments: Experiments

## Installation

```
git clone https://github.com/hp-transfer/ht_optimizers.git
git clone https://github.com/hp-transfer/ht_benchmarks.git
git clone https://github.com/hp-transfer/htaa_experiments.git
cd htaa_experiments

# Activate virtual / conda environment [-c conda-forge swig gcc_linux-64 gxx_linux-64]
poetry install
```



## Experiment results structure

```
<EXPERIMENT_GROUP>/                                        <<  Logical group of experiments
├── cluster_oe_<BENCHMARK>/                                <<  Output and error related to benchmark
├── <BENCHMARK>/                                           <<  Each benchmark gets own folder
│   └── <RUNTYPE>                                          <<  Each type of run gets own folder
│      └── <APPROACH>/                                     <<  Each approach gets own folder
│         └── trajectory_id_<ID>/                          <<  Each trajectory gets own folder
│            └── adjsutment_id_<ID>/                       <<  Each adjustment type gets own folder
│                └── repeat_<REPETITION>/                  <<  Each repetition gets own folder
│                    ├── .hydra/                           <<  Exact configuration files used in this run
│                    ├── eval/                             <<  Generalization results per step
│                    │   ├── batch_result_<STEP>.pkl
│                    ├── train/                            <<  Online results
│                    │   ├── batch_result_<STEP>.pkl
│                    │   └── trajectory.pkl
│                    └── <LOGGER>.log
└── <BENCHMARK>.args                                       <<  Argfile for benchmark
```


## License

[MIT](LICENSE)
