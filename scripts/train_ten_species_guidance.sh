#!/bin/bash
#SBATCH -o ../watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

<<comment
#  Usage:
cd scripts/
MODEL=<ar|mdlm|udlm>
sbatch \
  --export=ALL,MODEL=${MODEL} \
  --job-name=train_ten_species_guidance_${MODEL} \
  train_ten_species_guidance.sh
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
if [ -z "${USE_SIMPLE_CE_LOSS}" ]; then
  USE_SIMPLE_CE_LOSS=False
fi
RUN_NAME="${MODEL}"
if [ "${USE_SIMPLE_CE_LOSS}" = "True" ]; then
  RUN_NAME="${RUN_NAME}_simple-ce"
fi

if [ "${MODEL}" = "ar" ]; then
  # AR
  DIFFUSION="absorbing_state"
  PARAMETERIZATION="ar"
  T=0
  TIME_COND=False
  ZERO_RECON_LOSS=False
  BIDIRECTIONAL=False
  BIDIRECTIONAL_STRATEGY=null
  BIDIRECTIONAL_WEIGHT_TIE=null
elif [ "${MODEL}" = "mdlm" ]; then
  # MDLM
  DIFFUSION="absorbing_state"
  PARAMETERIZATION="subs"
  T=0
  TIME_COND=False
  ZERO_RECON_LOSS=False
  BIDIRECTIONAL=True
  BIDIRECTIONAL_STRATEGY=add
  BIDIRECTIONAL_WEIGHT_TIE=True
elif [ "${MODEL}" = "udlm" ]; then
  # UDLM
  DIFFUSION="uniform"
  PARAMETERIZATION="d3pm"
  T=0
  TIME_COND=True
  ZERO_RECON_LOSS=True
  BIDIRECTIONAL=True
  BIDIRECTIONAL_STRATEGY=add
  BIDIRECTIONAL_WEIGHT_TIE=True
else
  echo "MODEL must be one of ar, mdlm, udlm"
  exit 1
fi

# To enable preemption re-loading, set `hydra.run.dir` or
srun python -u -m main \
  diffusion=${DIFFUSION} \
  parameterization=${PARAMETERIZATION} \
  T=${T} \
  time_conditioning=${TIME_COND} \
  zero_recon_loss=${ZERO_RECON_LOSS} \
  data=ten_species \
  eval.generate_samples=False \
  loader.global_batch_size=32 \
  loader.eval_global_batch_size=64 \
  loader.batch_size=2 \
  backbone=dimamba \
  model=dimamba \
  model.bidirectional=${BIDIRECTIONAL} \
  model.bidirectional_strategy=${BIDIRECTIONAL_STRATEGY} \
  model.bidirectional_weight_tie=${BIDIRECTIONAL_WEIGHT_TIE} \
  model.length=32768 \
  optim.lr=2e-3 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=3000 \
  lr_scheduler.lr_min=2e-6 \
  training.guidance.cond_dropout=0.1 \
  training.compute_loss_on_pad_tokens=False \
  training.use_simple_ce_loss=${USE_SIMPLE_CE_LOSS} \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=6_000 \
  trainer.max_steps=30_000 \
  trainer.val_check_interval=3_000 \
  wandb.name="ten_species_${RUN_NAME}" \
  hydra.run.dir="./outputs/ten_species/${RUN_NAME}"
