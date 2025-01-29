DATA=$1
GPU=$2
EXP=$3

dirname="model_ckpt/${DATA}/exp_${EXP}/inference_carpk"
mkdir -p -- "$dirname"
python3 -m tools.test_carpk --config config_files/${DATA}.yaml \
					 		--gpus ${GPU} --exp ${EXP} --enc vpt --prompt single --ckpt_used 173_best \
							    | tee -a ${dirname}/log.txt



