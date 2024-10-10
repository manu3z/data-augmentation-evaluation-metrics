#!/bin/bash

# Run 'conda activate tf' on lennon before this script
# Run 'conda activate timegan' on local before this script

ORIDATADIR=src/data/original/stock_data.csv
GENDATADIR=src/data/generated/generated_data_1000e.npy
OUTDIR=timegan/stock
OUTFILENAME=stock
SEQLEN=24

mkdir -p -v "out/figures/${OUTDIR}"

# python src/histogram.py \
# 	-d ${ORIDATADIR} \
# 	-g ${GENDATADIR} \
# 	-o ${OUTDIR}/hist-${OUTFILENAME} \
# 	--seq_len ${SEQLEN} \
# 	-l

# python src/plot_data.py \
# 	-d ${ORIDATADIR} \
# 	-g ${GENDATADIR} \
# 	-o ${OUTDIR}/plot-${OUTFILENAME} \
# 	-s 2048

# python src/plot_data2.py \
# 	-d ${ORIDATADIR} \
# 	-g ${GENDATADIR} \
# 	-o ${OUTDIR}/plot2-${OUTFILENAME} \
# 	-s 2048

python src/plot_windows.py \
	-d ${ORIDATADIR} \
	-g ${GENDATADIR} \
	--seq_len ${SEQLEN} \
	-w 6 \
	-o ${OUTDIR}/plotwindows-${OUTFILENAME} \

# python src/PCA.py \
# 	-d ${ORIDATADIR} \
# 	-g ${GENDATADIR} \
# 	--seq_len ${SEQLEN} \
# 	-o ${OUTDIR}/PCA-${OUTFILENAME}
 
# python src/tSNE.py \
#     -d ${ORIDATADIR} \
#     -g ${GENDATADIR} \
# 	--seq_len ${SEQLEN} \
#     -o ${OUTDIR}/tSNE-${OUTFILENAME}

# python src/predictive.py \
# 	-d ${ORIDATADIR} \
# 	-g ${GENDATADIR} \
# 	--seq_len ${SEQLEN} \
# 	-i 10

# python src/discriminative.py \
# 	-d ${ORIDATADIR} \
# 	-g ${GENDATADIR} \
# 	--seq_len ${SEQLEN} \
# 	-i 10
