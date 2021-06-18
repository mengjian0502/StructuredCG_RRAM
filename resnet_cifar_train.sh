PYTHON="/home/mengjian/anaconda3/bin/python"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=resnet18_CG
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD

wd=0.0005
lr=0.1

# channel gating
cg_groups=2
cg_alpha=2.0
cg_threshold_init=0.0
cg_threshold_target=0.99
lambda_CG=1e-4
hard_sig=True
cg_grouping=False
cg_slide=True
slice_size=32

save_path="./save/strucCG/${model}/${model}_optim${optimizer}_lr${lr}_wd${wd}_cg${cg_groups}_cg_slide${cg_slide}_4bit/"
log_file="${model}_optim${optimizer}_lr${lr}_wd${wd}.log"

$PYTHON -W ignore main.py --dataset ${dataset} \
    --data_path ./dataset/ \
    --model ${model} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --schedule 60 120 \
    --gammas 0.1 0.1 \
    --batch_size ${batch_size} \
    --ngpu 1 \
    --wd ${wd} \
    --slice_size ${slice_size} \
    --cg_groups ${cg_groups} \
    --cg_alpha ${cg_alpha} \
    --cg_threshold_init ${cg_threshold_init} \
    --cg_threshold_target ${cg_threshold_target} \
    --lambda_CG ${lambda_CG} \
    --hard_sig ${hard_sig};