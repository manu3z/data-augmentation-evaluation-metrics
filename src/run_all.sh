#!/bin/bash

# Run 'conda activate tf' on lennon before this script
# Run 'conda activate timegan' on local before this script

ORIDATADIR=/home/msanchez/sparse-ts/data/TELCO_data_train_no-idx.csv
GENDATADIR=/home/msanchez/sparse-ts/timeVAE/outputs/gen_data/TELCO_data_train_no-idx_noperm_288_p4-m7-d288/timeVAE_TELCO_data_train_no-idx_noperm_288_prior_samples.npz
OUTDIR=timeVAE-p4-m7-d288/TELCO_noperm_288
OUTFILENAME=TELCO_noperm_288
SEQLEN=288

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

# python src/plot_windows.py \
# 	-d ${ORIDATADIR} \
# 	-g ${GENDATADIR} \
# 	--seq_len ${SEQLEN} \
# 	-w 5 \
# 	-o ${OUTDIR}/plotwindows-${OUTFILENAME} \

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

python src/predictive.py \
	-d ${ORIDATADIR} \
	-g ${GENDATADIR} \
	--seq_len ${SEQLEN} \
	-i 1

python src/discriminative.py \
	-d ${ORIDATADIR} \
	-g ${GENDATADIR} \
	--seq_len ${SEQLEN} \
	-i 1
