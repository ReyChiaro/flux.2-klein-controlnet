MODEL_PATH="black-forest-labs/FLUX.2-Klein-9B"
DATA_PATH="bucket.json"
DATA_ROOT="dataset"

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --gpu_ids 5 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    train_flux2klein_controlnet.py \
    --project-name train-flux2klein-controlnet \
    --output-dir outputs \
    --checkpoint-dir checkpoints \
    --evaluation-dir evalutaions \
    --log-dir logs \
    --base-model ${MODEL_PATH} \
    --load-text-encoder \
    --controlnet "" \
    --num-controlnet-layers 4 \
    --num-controlnet-single-layers 10 \
    --conditioning-scale 1.0 \
    --data-file ${DATA_PATH} \
    --data-root ${DATA_ROOT} \
    --batch-size 1 \
    --bucket-data \
    --num-workers 8 \
    --seed 42 \
    --max-training-steps 10 \
    --mixed-precision bf16 \
    --gradient-accumulation-steps 1 \
    --log-with tensorboard \
    --max-grad-norm 1.0 \
    --weighting-scheme logit_normal \
    --logit-mean 0.0 \
    --logit-std 1.0 \
    --mode-scale 1.29 \
    --save-steps 5 \
    --eval-steps 5 \
    --num-eval 5 \
    --gradient-checkpointing \