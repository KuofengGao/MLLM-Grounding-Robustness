CUDA_VISIBLE_DEVICES=0 python -m eval_scripts.untarget_text_attack \
                        --cfg-path "eval_configs/ut_minigptv2.yaml" \
                        --dataset refcoco \
                        --split val \
                        --step_size 0.0078 \
                        --epsilon 0.064 \
                        --iter 100
