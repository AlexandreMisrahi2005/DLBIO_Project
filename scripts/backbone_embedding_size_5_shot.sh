#!/bin/bash

echo "5-shot methods with 40 epochs, backbone sizes = [64, 32]"
python run.py dataset=tabula_muris method=matchingnet n_shot=5 exp.name=matchingnet_tabula_muris_5_shot_embedding_64_32 backbone.layer_dim=[64,32]
python run.py dataset=tabula_muris method=protonet n_shot=5 exp.name=protonet_tabula_muris_5_shot_embedding_64_32 backbone.layer_dim=[64,32]
python run.py dataset=tabula_muris method=maml n_shot=5 exp.name=maml_tabula_muris_5_shot_5_updates_innerlr_002_embedding_64_32 method.task_update_num=5 method.maml_inner_lr=0.002 backbone.layer_dim=[64,32]

echo "5-shot methods with 40 epochs, backbone sizes = [64, 128]"
python run.py dataset=tabula_muris method=matchingnet n_shot=5 exp.name=matchingnet_tabula_muris_5_shot_embedding_64_128 backbone.layer_dim=[64,128]
python run.py dataset=tabula_muris method=protonet n_shot=5 exp.name=protonet_tabula_muris_5_shot_embedding_64_128 backbone.layer_dim=[64,128]
python run.py dataset=tabula_muris method=maml n_shot=5 exp.name=maml_tabula_muris_5_shot_5_updates_innerlr_002_embedding_64_128 method.task_update_num=5 method.maml_inner_lr=0.002 backbone.layer_dim=[64,128]

echo "5-shot metaoptnet-svm with 40 epochs, backbone sizes = [64, 32], varying C_reg"
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_10_maxiter_3_embedding_64_32 method.base_learner=svm method.C_reg=10 backbone.layer_dim=[64,32]
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_5_maxiter_3_embedding_64_32 method.base_learner=svm method.C_reg=5 backbone.layer_dim=[64,32]
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_1_maxiter_3_embedding_64_32 method.base_learner=svm method.C_reg=1 backbone.layer_dim=[64,32]

echo "5-shot metaoptnet-svm with 40 epochs, backbone sizes = [64, 128], varying C_reg"
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_100_maxiter_3_embedding_64_128 method.base_learner=svm method.C_reg=100 backbone.layer_dim=[64,128]
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_50_maxiter_3_embedding_64_128 method.base_learner=svm method.C_reg=50 backbone.layer_dim=[64,128]
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_20_maxiter_3_embedding_64_128 method.base_learner=svm method.C_reg=20 backbone.layer_dim=[64,128]
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_10_maxiter_3_embedding_64_128 method.base_learner=svm method.C_reg=10 backbone.layer_dim=[64,128]
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_5_maxiter_3_embedding_64_128 method.base_learner=svm method.C_reg=5 backbone.layer_dim=[64,128]
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_svm_tabula_muris_5_shot_C_reg_1_maxiter_3_embedding_64_128 method.base_learner=svm method.C_reg=1 backbone.layer_dim=[64,128]

echo "5-shot metaoptnet-ridge with 40 epochs, backbone sizes = [64, 32], varying lambda_reg"
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_ridge_tabula_muris_5_shot_lambda_reg_10_embedding_64_32 method.base_learner=ridge method.lambda_reg=10 backbone.layer_dim=[64,32]
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_ridge_tabula_muris_5_shot_lambda_reg_5_embedding_64_32 method.base_learner=ridge method.lambda_reg=5 backbone.layer_dim=[64,32]
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_ridge_tabula_muris_5_shot_lambda_reg_1_embedding_64_32 method.base_learner=ridge method.lambda_reg=1 backbone.layer_dim=[64,32]

echo "5-shot metaoptnet-ridge with 40 epochs, backbone sizes = [64, 128], varying lambda_reg"
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_ridge_tabula_muris_5_shot_lambda_reg_10_embedding_64_128 method.base_learner=ridge method.lambda_reg=10 backbone.layer_dim=[64,128]
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_ridge_tabula_muris_5_shot_lambda_reg_5_embedding_64_128 method.base_learner=ridge method.lambda_reg=5 backbone.layer_dim=[64,128]
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_ridge_tabula_muris_5_shot_lambda_reg_1_embedding_64_128 method.base_learner=ridge method.lambda_reg=1 backbone.layer_dim=[64,128]
python run.py dataset=tabula_muris method=metaoptnet n_shot=5 exp.name=metaoptnet_ridge_tabula_muris_5_shot_lambda_reg_01_embedding_64_128 method.base_learner=ridge method.lambda_reg=0.1 backbone.layer_dim=[64,128]
