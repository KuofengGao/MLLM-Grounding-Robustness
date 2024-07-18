CUDA_VISIBLE_DEVICES=0 python -m eval_scripts.target_to_all \
                        --cfg-path "eval_configs/ta_minigptv2.yaml" \
                        --dataset refcoco \
                        --split val \
                        --step_size 0.0078 \
                        --epsilon 0.064 \
                        --iter 100
