# Hyperparameter Transfer Across Developer Adjustments: Experiments


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
