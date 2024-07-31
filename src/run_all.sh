#!/bin/bash

# Run 'conda activate timegan' before this script

ORIDATADIR=src/data/original/stock_data.csv
GENDATADIR=src/data/generated/gen_stock_data.npy
OUTDIR=experiment1
OUTFILENAME=exp1

mkdir -p -v "out/figures/${OUTDIR}"

python src/histogram.py \
	-d ${ORIDATADIR} \
	-g ${GENDATADIR} \
	-o ${OUTDIR}/hist-${OUTFILENAME} \
	-l

python src/plot_data2.py \
	-d ${ORIDATADIR} \
	-g ${GENDATADIR} \
	-o ${OUTDIR}/plot-${OUTFILENAME} \
	-s 2000

python src/PCA.py \
	-d ${ORIDATADIR} \
	-g ${GENDATADIR} \
	-o ${OUTDIR}/PCA-${OUTFILENAME}
 
python src/tSNE.py \
    -d ${ORIDATADIR} \
    -g ${GENDATADIR} \
    -o ${OUTDIR}/tSNE-${OUTFILENAME}

python src/predictive.py \
	-d ${ORIDATADIR} \
	-g ${GENDATADIR} \
	-i 10

python src/discriminative.py \
	-d ${ORIDATADIR} \
	-g ${GENDATADIR} \
	-i 10
