python train.py -m -cp=conf/big5/ -cn=startup hydra/launcher=joblib hydra.launcher.n_jobs=8 jafa_cfg.r_cost='range(-0.2,-0.049,0.01)' jafa_cfg.pretrain=5000 &&\
python train.py -m -cp=conf/cube/ -cn=startup hydra/launcher=joblib hydra.launcher.n_jobs=100 jafa_cfg.r_cost='range(-0.51,-0.01,0.02)' &&\
python train.py -m -cp=conf/grid -cn=startup hydra/launcher=joblib hydra.launcher.n_jobs=100 jafa_cfg.r_cost='range(-0.61,-0.01,0.02)' &&\
python train.py -m -cp=conf/gas -cn=startup hydra/launcher=joblib hydra.launcher.n_jobs=3 jafa_cfg.r_cost='range(-0.61,-0.01,0.1)'


python train.py -m -cp=conf/big5/ -cn=startup hydra/launcher=joblib hydra.launcher.n_jobs=8 jafa_cfg.r_cost='range(-0.002,-0.000249,0.00025)' jafa_cfg.pretrain=500 &&\
python train.py -m -cp=conf/cube/ -cn=startup hydra/launcher=joblib hydra.launcher.n_jobs=100 jafa_cfg.r_cost='range(-0.002,-0.000249,0.00025)' jafa_cfg.pretrain=5000 &&\
python train.py -m -cp=conf/grid -cn=startup hydra/launcher=joblib hydra.launcher.n_jobs=100 jafa_cfg.r_cost='range(-0.002,-0.000249,0.00025)' jafa_cfg.pretrain=500 &&\
python train.py -m -cp=conf/gas -cn=startup hydra/launcher=joblib hydra.launcher.n_jobs=3 jafa_cfg.r_cost='range(-0.005,-0.000249,0.00025)' jafa_cfg.pretrain=500

python train.py -m -cp=conf/mnist/ -cn=startup hydra/launcher=joblib hydra.launcher.n_jobs=1 jafa_cfg.r_cost='range(-0.005,-0.000249,0.00025)' jafa_cfg.pretrain=500 # 33GB of gpu

python train.py -m -cp=conf/fashion/ -cn=startup hydra/launcher=joblib hydra.launcher.n_jobs=1 jafa_cfg.r_cost='range(-0.005,-0.000249,0.00025)' jafa_cfg.pretrain=500 # 33GB of gpu
