#!/bin/bash

# Runs with other scope
echo "Starting SVM training runs for other scope"
python run.py method=metaoptnet method.base_learner=svm method.C_reg=20 dataset=tabula_muris n_shot=1 method.stop_epoch=60 exp.name=svm_C_20_iter_3_tabula_muris_1_shot
python run.py method=metaoptnet method.base_learner=svm method.C_reg=20 dataset=swissprot n_shot=5 method.stop_epoch=40 exp.name=svm_C_20_iter_3_swissprot_5_shot

# SVM training runs for Swiss Prot dataset
echo "Starting SVM training runs for Swiss Prot dataset"

# 5-shot, 40 epochs, various C_reg and maxIter configurations
python run.py method=metaoptnet method.base_learner=svm method.C_reg=10 method.maxIter=1 dataset=swissprot n_shot=5 method.stop_epoch=40 exp.name=svm_C_10_iter_1_swissprot_5_shot
python run.py method=metaoptnet method.base_learner=svm method.C_reg=5 method.maxIter=1 dataset=swissprot n_shot=5 method.stop_epoch=40 exp.name=svm_C_5_iter_1_swissprot_5_shot
python run.py method=metaoptnet method.base_learner=svm method.C_reg=1 method.maxIter=1 dataset=swissprot n_shot=5 method.stop_epoch=40 exp.name=svm_C_1_iter_1_swissprot_5_shot

# 1-shot, 60 epochs, various C_reg and maxIter configurations
python run.py method=metaoptnet method.base_learner=svm method.C_reg=10 method.maxIter=1 dataset=swissprot n_shot=1 method.stop_epoch=60 exp.name=svm_C_10_iter_1_swissprot_1_shot
python run.py method=metaoptnet method.base_learner=svm method.C_reg=5 method.maxIter=1 dataset=swissprot n_shot=1 method.stop_epoch=60 exp.name=svm_C_5_iter_1_swissprot_1_shot
python run.py method=metaoptnet method.base_learner=svm method.C_reg=1 method.maxIter=1 dataset=swissprot n_shot=1 method.stop_epoch=60 exp.name=svm_C_1_iter_1_swissprot_1_shot

# 5-shot, 40 epochs, maxIter increased to 10
python run.py method=metaoptnet method.base_learner=svm method.C_reg=10 method.maxIter=10 dataset=swissprot n_shot=5 method.stop_epoch=40 exp.name=svm_C_10_iter_10_swissprot_5_shot
python run.py method=metaoptnet method.base_learner=svm method.C_reg=5 method.maxIter=10 dataset=swissprot n_shot=5 method.stop_epoch=40 exp.name=svm_C_5_iter_10_swissprot_5_shot
python run.py method=metaoptnet method.base_learner=svm method.C_reg=1 method.maxIter=10 dataset=swissprot n_shot=5 method.stop_epoch=40 exp.name=svm_C_1_iter_10_swissprot_5_shot

# 1-shot, 60 epochs, maxIter increased to 10
python run.py method=metaoptnet method.base_learner=svm method.C_reg=10 method.maxIter=10 dataset=swissprot n_shot=1 method.stop_epoch=60 exp.name=svm_C_10_iter_10_swissprot_1_shot
python run.py method=metaoptnet method.base_learner=svm method.C_reg=5 method.maxIter=10 dataset=swissprot n_shot=1 method.stop_epoch=60 exp.name=svm_C_5_iter_10_swissprot_1_shot
python run.py method=metaoptnet method.base_learner=svm method.C_reg=1 method.maxIter=10 dataset=swissprot n_shot=1 method.stop_epoch=60 exp.name=svm_C_1_iter_10_swissprot_1_shot

echo "All SVM training runs for Swiss Prot dataset are initiated"

