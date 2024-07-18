CUDA_VISIBLE_DEVICES=0 python -m eval_scripts.untarget_embedding_attack \
                        --cfg-path "eval_configs/ue_minigptv2.yaml" \
                        --dataset refcoco \
                        --split val \
                        --step_size 0.0078 \
                        --epsilon 0.064 \
                        --iter 100

CUDA_VISIBLE_DEVICES=0 python -m eval_scripts.untarget_embedding_attack \
                        --cfg-path "eval_configs/ue_minigptv2.yaml" \
                        --dataset refcoco \
                        --split testA \
                        --step_size 0.0078 \
                        --epsilon 0.064 \
                        --iter 100

CUDA_VISIBLE_DEVICES=0 python -m eval_scripts.untarget_embedding_attack \
                        --cfg-path "eval_configs/ue_minigptv2.yaml" \
                        --dataset refcoco \
                        --split testB \
                        --step_size 0.0078 \
                        --epsilon 0.064 \
                        --iter 100

CUDA_VISIBLE_DEVICES=0 python -m eval_scripts.untarget_embedding_attack \
                        --cfg-path "eval_configs/ue_minigptv2.yaml" \
                        --dataset refcocog \
                        --split val \
                        --step_size 0.0078 \
                        --epsilon 0.064 \
                        --iter 100

CUDA_VISIBLE_DEVICES=0 python -m eval_scripts.untarget_embedding_attack \
                        --cfg-path "eval_configs/ue_minigptv2.yaml" \
                        --dataset refcoco+ \
                        --split val \
                        --step_size 0.0078 \
                        --epsilon 0.064 \
                        --iter 100

CUDA_VISIBLE_DEVICES=0 python -m eval_scripts.untarget_embedding_attack \
                        --cfg-path "eval_configs/ue_minigptv2.yaml" \
                        --dataset refcocog \
                        --split test \
                        --step_size 0.0078 \
                        --epsilon 0.064 \
                        --iter 100

CUDA_VISIBLE_DEVICES=0 python -m eval_scripts.untarget_embedding_attack \
                        --cfg-path "eval_configs/ue_minigptv2.yaml" \
                        --dataset refcoco+ \
                        --split testA \
                        --step_size 0.0078 \
                        --epsilon 0.064 \
                        --iter 100

CUDA_VISIBLE_DEVICES=0 python -m eval_scripts.untarget_embedding_attack \
                        --cfg-path "eval_configs/ue_minigptv2.yaml" \
                        --dataset refcoco+ \
                        --split testB \
                        --step_size 0.0078 \
                        --epsilon 0.064 \
                        --iter 100

CUDA_VISIBLE_DEVICES=0 python -m eval_scripts.untarget_embedding_attack \
                        --cfg-path "eval_configs/ue_minigptv2.yaml" \
                        --dataset refcocog \
                        --split val \
                        --step_size 0.0078 \
                        --epsilon 0.064 \
                        --iter 100

CUDA_VISIBLE_DEVICES=0 python -m eval_scripts.untarget_embedding_attack \
                        --cfg-path "eval_configs/ue_minigptv2.yaml" \
                        --dataset refcocog \
                        --split test \
                        --step_size 0.0078 \
                        --epsilon 0.064 \
                        --iter 100
