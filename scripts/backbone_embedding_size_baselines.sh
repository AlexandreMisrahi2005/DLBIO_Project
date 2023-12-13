#!/bin/bash

echo "Swiss Prot"
python run.py method=metaoptnet method.base_learner=svm method.C_reg=20 method.maxIter=1 dataset=swissprot n_shot=5 method.stop_epoch=40 exp.name=svm_C_20_iter_1_swissprot_5_shot
python run.py method=metaoptnet method.base_learner=svm method.C_reg=50 method.maxIter=1 dataset=swissprot n_shot=5 method.stop_epoch=40 exp.name=svm_C_50_iter_1_swissprot_5_shot

python run.py method=metaoptnet method.base_learner=svm method.C_reg=20 method.maxIter=1 dataset=swissprot n_shot=1 method.stop_epoch=60 exp.name=svm_C_20_iter_1_swissprot_1_shot
python run.py method=metaoptnet method.base_learner=svm method.C_reg=50 method.maxIter=1 dataset=swissprot n_shot=1 method.stop_epoch=60 exp.name=svm_C_50_iter_1_swissprot_1_shot

python run.py method=metaoptnet method.base_learner=svm method.C_reg=0.5 method.maxIter=10 dataset=swissprot n_shot=1 method.stop_epoch=60 exp.name=svm_C_05_iter_10_swissprot_1_shot
python run.py method=metaoptnet method.base_learner=svm method.C_reg=0.1 method.maxIter=10 dataset=swissprot n_shot=1 method.stop_epoch=60 exp.name=svm_C_01_iter_10_swissprot_1_shot

echo "TABULA MURIS"
echo "backbone sizes = [64, 32]"
echo "5-shot with 40 epochs"
python run.py dataset=tabula_muris method=baseline n_shot=5 exp.name=baseline_tabula_muris_5_shot_embedding_64_32 backbone.layer_dim=[64,32]
python run.py dataset=tabula_muris method=baseline_pp n_shot=5 exp.name=baseline_pp_tabula_muris_5_shot_embedding_64_32 backbone.layer_dim=[64,32]

echo "1-shot with 60 epochs"
python run.py dataset=tabula_muris method=baseline n_shot=1 exp.name=baseline_tabula_muris_1_shot_embedding_64_32 backbone.layer_dim=[64,32]
python run.py dataset=tabula_muris method=baseline_pp n_shot=1 exp.name=baseline_pp_tabula_muris_1_shot_embedding_64_32 backbone.layer_dim=[64,32]

echo "backbone sizes = [64, 128]"
echo "5-shot with 40 epochs"
python run.py dataset=tabula_muris method=baseline n_shot=5 exp.name=baseline_tabula_muris_5_shot_embedding_64_128 backbone.layer_dim=[64,128]
python run.py dataset=tabula_muris method=baseline_pp n_shot=5 exp.name=baseline_pp_tabula_muris_5_shot_embedding_64_128 backbone.layer_dim=[64,128]

echo "1-shot with 60 epochs"
python run.py dataset=tabula_muris method=baseline n_shot=1 exp.name=baseline_tabula_muris_1_shot_embedding_64_128 backbone.layer_dim=[64,128]
python run.py dataset=tabula_muris method=baseline_pp n_shot=1 exp.name=baseline_pp_tabula_muris_1_shot_embedding_64_128 backbone.layer_dim=[64,128]

echo "SWISS PROT"
echo "backbone sizes = [512, 256]"
echo "5-shot with 40 epochs"
python run.py dataset=swissprot method=baseline n_shot=5 exp.name=baseline_swissprot_5_shot_embedding_512_256 backbone.layer_dim=[512,256]
python run.py dataset=swissprot method=baseline_pp n_shot=5 exp.name=baseline_pp_swissprot_5_shot_embedding_512_256 backbone.layer_dim=[512,256]

echo "1-shot with 60 epochs"
python run.py dataset=swissprot method=baseline n_shot=1 exp.name=baseline_swissprot_1_shot_embedding_512_256 backbone.layer_dim=[512,256]
python run.py dataset=swissprot method=baseline_pp n_shot=1 exp.name=baseline_pp_swissprot_1_shot_embedding_512_256 backbone.layer_dim=[512,256]

echo "backbone sizes = [512, 1024]"
echo "5-shot with 40 epochs"
python run.py dataset=swissprot method=baseline n_shot=5 exp.name=baseline_swissprot_5_shot_embedding_512_1024 backbone.layer_dim=[512,1024]
python run.py dataset=swissprot method=baseline_pp n_shot=5 exp.name=baseline_pp_swissprot_5_shot_embedding_512_1024 backbone.layer_dim=[512,1024]

echo "1-shot with 60 epochs"
python run.py dataset=swissprot method=baseline n_shot=1 exp.name=baseline_swissprot_1_shot_embedding_512_1024 backbone.layer_dim=[512,1024]
python run.py dataset=swissprot method=baseline_pp n_shot=1 exp.name=baseline_pp_swissprot_1_shot_embedding_512_1024 backbone.layer_dim=[512,1024]

