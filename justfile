# List available receipes
@list:
  just --list

# Sync result data from cluster
get experiment_group host="meta":
  rsync -a --info=progress2 --info=name0 {{host}}:htaa_experiments/results/{{experiment_group}} results/
  rm results/{{experiment_group}}/_load_cache.csv

# Plot a result
plot experiment_group output_dir="plots":
  mkdir -p {{output_dir}}/{{experiment_group}}
  python -m hp_transfer_aa_experiments.analyse.plot_speedup \
    --results_path results/{{experiment_group}} \
    --output_dir {{output_dir}}/{{experiment_group}}
  python -m hp_transfer_aa_experiments.analyse.plot_failure_percent \
    --results_path results/{{experiment_group}} \
    --output_dir {{output_dir}}/{{experiment_group}}
  python -m hp_transfer_aa_experiments.analyse.plot_improvements \
    --results_path results/{{experiment_group}} \
    --output_dir {{output_dir}}/{{experiment_group}}

# Plot a result and open
ploto experiment_group output_dir="plots":
  just plot {{experiment_group}} {{output_dir}}
  okular {{output_dir}}/{{experiment_group}}/speedup_aggregate.pdf &

# Get results from cluster and plot
getplot experiment_group host="meta" output_dir="plots":
  just get {{experiment_group}} {{host}}
  just plot {{experiment_group}} {{output_dir}}

  # Get results from cluster and plot
getploto experiment_group host="meta" output_dir="plots":
  just getplot {{experiment_group}} {{output_dir}} {{host}}
  okular {{output_dir}}/speedup_aggregate.pdf &

fill from to approach="tpe" runtype="eval_reference":
  mkdir -p results/{{to}}/fcnet_aa/{{runtype}}/{{approach}}/
  mkdir -p results/{{to}}/xgb_aa/{{runtype}}/{{approach}}/
  mkdir -p results/{{to}}/svm_aa/{{runtype}}/{{approach}}/
  cp results/{{from}}/fcnet_aa/{{runtype}}/{{approach}} results/{{to}}/fcnet_aa/{{runtype}}/ -r
  cp results/{{from}}/svm_aa/{{runtype}}/{{approach}} results/{{to}}/svm_aa/{{runtype}}/ -r
  cp results/{{from}}/xgb_aa/{{runtype}}/{{approach}} results/{{to}}/xgb_aa/{{runtype}}/ -r
  rm results/{{to}}/_load_cache.csv

# Generate argfile and submit job
submit experiment_group benchmark runtypes approaches num_workers trajectory_ids adjustment_ids queue repeats nic_name max_jobs offset="0" cluster="meta":
  #!/bin/bash
  set -e  # Stop on first failure

  EXPERIMENT_GROUP_DIR=results/{{experiment_group}}
  if test -e ${EXPERIMENT_GROUP_DIR}; then
    echo "WARNING: ${EXPERIMENT_GROUP_DIR} already exists"
  fi

  # Generate argfile
  ARGS_FILE_FOLDER=${EXPERIMENT_GROUP_DIR}/args
  mkdir -p ${ARGS_FILE_FOLDER}
  NUM_BENCHMARK_ARGFILES=$(ls ${ARGS_FILE_FOLDER} | wc -l)
  ARGS_FILE=${ARGS_FILE_FOLDER}/{{benchmark}}_${NUM_BENCHMARK_ARGFILES}.args

  python submit/gen_args_file.py \
    --experiment_group {{experiment_group}} \
    --benchmark {{benchmark}} \
    --approaches '{{approaches}}' \
    --repeats {{repeats}} \
    --num_workers {{num_workers}} \
    --trajectory_ids '{{trajectory_ids}}' \
    --adjustment_ids '{{adjustment_ids}}' \
    --runtypes '{{runtypes}}' \
    --argfile ${ARGS_FILE} \
    --offset_repetition {{offset}} \
    --nic_name {{nic_name}}

  # Create error and output directory for benchmark
  CLUSTER_OE=${EXPERIMENT_GROUP_DIR}/cluster_oe_{{benchmark}}
  mkdir -p ${CLUSTER_OE}

  # Count lines in argfile to infer number of arrays
  NUM_ARRAYS=$(grep -c "^" $ARGS_FILE)

  # Submit
  if [ {{cluster}} = "nemo" ]; then
    msub \
      -o ${CLUSTER_OE}/test.out \
      -e ${CLUSTER_OE}/test.err \
      -t 1-${NUM_ARRAYS} \
      -v ARGS_FILE=${ARGS_FILE} \
      submit/{{benchmark}}_nemo.sh
  else
    sbatch \
      -o ${CLUSTER_OE}/%x.oe \
      -e ${CLUSTER_OE}/%x.oe \
      -p {{queue}} \
      --array=1-${NUM_ARRAYS}%{{max_jobs}} \
      --export=ARGS_FILE=${ARGS_FILE} \
      submit/{{benchmark}}.sh
  fi
