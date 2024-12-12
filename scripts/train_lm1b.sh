#!/bin/bash
#SBATCH -o ../watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

<<comment
#  Usage:
cd scripts/
MODEL=<ar|mdlm|udlm>
sbatch \
  --export=ALL,MODEL=${MODEL} \
  --job-name=train_lm1b_pad_${MODEL} \
  train_lm1b_pad.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1

# Expecting:
#  - MODEL (ar, mdlm, udlm)
if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi
RUN_NAME="${MODEL}_fix-ema"

if [ "${MODEL}" = "ar" ]; then
  # AR
  DIFFUSION="absorbing_state"
  PARAMETERIZATION="ar"
  T=0
  TIME_COND=False
  ZERO_RECON_LOSS=False
  sampling_use_cache=False
elif [ "${MODEL}" = "mdlm" ]; then
  # MDLM
  DIFFUSION="absorbing_state"
  PARAMETERIZATION="subs"
  T=0
  TIME_COND=False
  ZERO_RECON_LOSS=False
  sampling_use_cache=True
elif [ "${MODEL}" = "udlm" ]; then
  # UDLM
  DIFFUSION="uniform"
  PARAMETERIZATION="d3pm"
  T=0
  TIME_COND=True
  ZERO_RECON_LOSS=True
  sampling_use_cache=False
else
  echo "MODEL must be one of ar, mdlm, udlm"
  exit 1
fi


# To enable preemption re-loading, set `hydra.run.dir` or
# `checkpointing.save_dir` explicitly.
srun python -u -m main \
  diffusion="${DIFFUSION}" \
  parameterization="${PARAMETERIZATION}" \
  T=${T} \
  time_conditioning=${TIME_COND} \
  zero_recon_loss=${ZERO_RECON_LOSS} \
  data="lm1b" \
  data.wrap=False \
  data.tokenizer_name_or_path=bert-base-uncased \
  loader.global_batch_size=512 \
  loader.eval_global_batch_size=1024 \
  loader.batch_size=64 \
  loader.eval_batch_size=128 \
  backbone="dit" \
  model=small \
  model.length=128 \
  optim.lr=3e-4 \
  training.guidance=null \
  training.compute_loss_on_pad_tokens=False \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=100_000 \
  trainer.log_every_n_steps=100 \
  trainer.max_steps=1_000_000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=10_000 \
  eval.generate_samples=True \
  sampling.num_sample_batches=1 \
  sampling.batch_size=2 \
  sampling.use_cache=${sampling_use_cache} \
  sampling.steps=128 \
  wandb.name="lm1b_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/lm1b/${RUN_NAME}"
