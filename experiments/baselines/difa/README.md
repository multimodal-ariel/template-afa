## DIFA
### pretain vaeac for difa
```shell
python train.py -m -cp=conf/big5/ -cn=vaeac hydra/launcher=joblib hydra.launcher.n_jobs=16 difa_cfg.acquisition_cost='range(-0.002,-0.000249,0.00025)' &&\
python train.py -m -cp=conf/cube/ -cn=vaeac hydra/launcher=joblib hydra.launcher.n_jobs=100 difa_cfg.acquisition_cost='range(-0.002,-0.000249,0.00025)' &&\
python train.py -m -cp=conf/grid -cn=vaeac hydra/launcher=joblib hydra.launcher.n_jobs=100 difa_cfg.acquisition_cost='range(-0.002,-0.000249,0.00025)' &&\
python train.py -m -cp=conf/gas -cn=vaeac hydra/launcher=joblib hydra.launcher.n_jobs=6 difa_cfg.acquisition_cost='range(-0.005,-0.000249,0.00025)'
```
### train difa
```shell
python train.py -m -cp=conf/big5/ -cn=difa hydra/launcher=joblib hydra.launcher.n_jobs=16 +imputation_model_cfg.run_id='range(0,8,1)' &&\
python train.py -m -cp=conf/cube/ -cn=difa hydra/launcher=joblib hydra.launcher.n_jobs=100 +imputation_model_cfg.run_id='range(0,8,1)' &&\
python train.py -m -cp=conf/grid -cn=difa hydra/launcher=joblib hydra.launcher.n_jobs=100 +imputation_model_cfg.run_id='range(0,8,1)' &&\
python train.py -m -cp=conf/gas -cn=difa hydra/launcher=joblib hydra.launcher.n_jobs=20 +imputation_model_cfg.run_id='range(0,20,1)'
```