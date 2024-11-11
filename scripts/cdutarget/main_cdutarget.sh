#!/bin/bash

# custom config
DATA=/Workplace_sdb/dxw/data # your directory

DATASET=$1
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
NTOK=$5
DOMAINS=$6
GPU=$7
KD=$8

LOCATION=middle
DEEP=False
DEEPLAYER=None
SHOTS=0



DIR=output/cdutarget/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/deep${DEEP}_${LOCATION}/kd${KD}_${DOMAINS}_ntok${NTOK}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}, so skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
        --gpu ${GPU} \
        --kd ${KD} \
        --backbone ${BACKBONE} \
        --domains ${DOMAINS} \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.CDUTARGET.NUM_TOKENS ${NTOK} \
        TRAINER.CDUTARGET.N_CTX ${NTOK} \
        TRAINER.CDUTARGET.T_DEEP ${DEEP} \
        TRAINER.CDUTARGET.V_DEEP ${DEEP} \
        TRAINER.CDUTARGET.LOCATION ${LOCATION} \
        TRAINER.CDUTARGET.DEEP_LAYERS ${DEEPLAYER} \
        DATASET.NUM_SHOTS ${SHOTS}
fi
