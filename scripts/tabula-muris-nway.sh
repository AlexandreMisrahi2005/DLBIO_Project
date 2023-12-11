#!/bin/bash

echo "Running experiment for n_way = 20"

echo "matchingnet + protonet"
python run.py dataset=tabula_muris method=matchingnet n_shot=5 n_way=20 exp.name=matchingnet_tabula_muris_5_shot_20_way
python run.py dataset=tabula_muris method=protonet n_shot=5 n_way=20 exp.name=protonet_tabula_muris_5_shot_20_way

echo "MAML"
python run.py dataset=tabula_muris method=maml n_shot=5 n_way=20 exp.name=maml_tabula_muris_5_shot_1_update_20_way method.task_update_num=1 method.maml_inner_lr=0.01 method.stop_epoch=40
python run.py dataset=tabula_muris method=maml n_shot=5 n_way=20 exp.name=maml_tabula_muris_5_shot_5_updates_innerlr_002_20_way method.task_update_num=5 method.maml_inner_lr=0.002 method.stop_epoch=40

echo "metaoptnet-svm with varying C_reg"
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 n_way=20 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_50_maxiter_3_20_way method.base_learner=svm method.C_reg=50
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 n_way=20 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_20_maxiter_3_20_way method.base_learner=svm method.C_reg=20
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 n_way=20 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_10_maxiter_3_20_way method.base_learner=svm method.C_reg=10
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 n_way=20 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_5_maxiter_3_20_way method.base_learner=svm method.C_reg=5
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 n_way=20 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_1_maxiter_3_20_way method.base_learner=svm method.C_reg=1
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 n_way=20 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_01_maxiter_3_20_way method.base_learner=svm method.C_reg=0.1

# baselines
echo "baselines"
python run.py dataset=tabula_muris method=baseline n_shot=5 n_way=20 exp.name=baseline_tabula_muris_5_shot_20_way method.stop_epoch=40
python run.py dataset=tabula_muris method=baseline_pp n_shot=5 n_way=20 exp.name=baseline_pp_tabula_muris_5_shot_20_way method.stop_epoch=40
