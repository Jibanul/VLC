DATA=$1
GPU=$2
EXP=$3

dirname="model_ckpt/${DATA}/exp_${EXP}"
mkdir -p -- "$dirname"

# run in terminal
# sh scripts/train.sh FSC 0 4


# v5_3_1: ignore vls loss
python3 -m tools.train --config config_files/${DATA}.yaml \
					 		--gpus ${GPU} --exp ${EXP} --enc vpt --num_tokens 10 --patch_size 16 --prompt single \
							    | tee -a ${dirname}/log.txt