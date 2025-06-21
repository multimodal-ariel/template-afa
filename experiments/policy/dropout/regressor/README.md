`python train.py -m -cp=conf/cube/ -cn=startup hydra/launcher=joblib hydra.launcher.n_jobs=100 +mktmpl_exp.exp_p='experiments/make_template/outputs/cube/20250305_155142' +mktmpl_exp.run_id='range(0,16)'`

on leela
```shell
python train.py -m -cp=conf/gas/ -cn=startup hydra/launcher=joblib +mktmpl_exp.exp_p=experiments/make_template/outputs/gas_cnnet/20250324_224734 +mktmpl_exp.run_id='range(0,62)' hydra.launcher.n_jobs=100
python train.py -m -cp=conf/mnist/ -cn=startup hydra/launcher=joblib +mktmpl_exp.exp_p=experiments/make_template/outputs/mnist_cnnet/20250326_003820 +mktmpl_exp.run_id='range(0,37)' hydra.launcher.n_jobs=70
```