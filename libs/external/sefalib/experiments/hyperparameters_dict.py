"""Hyperparameters for the main training are kept here."""

hyperparameters_dict = {
    #############################################################
    #############################################################
    ######################     Invase 4       ###################
    #############################################################
    #############################################################
    "invase_4": {
        # RL hyperparameters
        "opportunistic": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "gamma": 0.5,
            "num_samples_predict": 50,
            "num_episodes": 20000,
            "lr": 0.001,
            "batchsize": 128,
            "print_every": 500,
            "eval_every": 100,
        },
        # DIME hyperparameters
        "dime": {
            "hidden_dim": 100,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 2,
            "eps_scale": 0.2,
        },
        # GDFS hyperparameters
        "gdfs": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 512,
            "patience": 2,
            "temp_scale": 0.1,
        },
        # Fixed MLP hyperparameters
        "fixed_mlp": {
            "hidden_dim": 250,
            "num_hidden": 3,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 256,
            "patience": 10,
        },
        # VAE hyperparameters
        "vae": {
            "latent_dim": 30,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 100,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 100,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 100,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Eddi hyperparameters
        "eddi": {
            "c_dim": 50,
            "latent_dim": 100,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 200,
            "epochs": 400,
            "lr": 0.001,
            "batchsize": 128,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Our Hyper parameters
        "ours": {
            "latent_dim": 6,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 150,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 50,
            "num_samples_train": 100,
            "num_samples_predict": 200,
            "num_samples_acquire": 200,
            "ib_beta": 0.0005,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
        },
    },
    #############################################################
    #############################################################
    ######################     Invase 5       ###################
    #############################################################
    #############################################################
    "invase_5": {
        # RL hyperparameters
        "opportunistic": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "gamma": 0.5,
            "num_samples_predict": 50,
            "num_episodes": 20000,
            "lr": 0.001,
            "batchsize": 128,
            "print_every": 500,
            "eval_every": 100,
        },
        # DIME hyperparameters
        "dime": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
            "eps_scale": 0.1,
        },
        # GDFS hyperparameters
        "gdfs": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 2,
            "temp_scale": 0.1,
        },
        # Fixed MLP hyperparameters
        "fixed_mlp": {
            "hidden_dim": 250,
            "num_hidden": 3,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 256,
            "patience": 10,
        },
        # VAE hyperparameters
        "vae": {
            "latent_dim": 10,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 200,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 100,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Eddi hyperparameters
        "eddi": {
            "c_dim": 20,
            "latent_dim": 20,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 200,
            "epochs": 400,
            "lr": 0.001,
            "batchsize": 512,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Our hyperparameters
        "ours": {
            "latent_dim": 4,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 100,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 20,
            "num_samples_train": 100,
            "num_samples_predict": 200,
            "num_samples_acquire": 200,
            "ib_beta": 0.0005,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
        },
    },
    #############################################################
    #############################################################
    ######################     Invase 6       ###################
    #############################################################
    #############################################################
    "invase_6": {
        # RL hyperparameters
        "opportunistic": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "gamma": 0.75,
            "num_samples_predict": 50,
            "num_episodes": 20000,
            "lr": 0.001,
            "batchsize": 256,
            "print_every": 500,
            "eval_every": 100,
        },
        # DIME hyperparameters
        "dime": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
            "eps_scale": 0.4,
        },
        # GDFS hyperparameters
        "gdfs": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 2,
            "temp_scale": 0.1,
        },
        # Fixed MLP hyperparameters
        "fixed_mlp": {
            "hidden_dim": 250,
            "num_hidden": 3,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 256,
            "patience": 10,
        },
        # VAE hyperparameters
        "vae": {
            "latent_dim": 40,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 200,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 200,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 3,
        },
        # Eddi hyperparameters
        "eddi": {
            "c_dim": 50,
            "latent_dim": 100,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 200,
            "epochs": 400,
            "lr": 0.001,
            "batchsize": 128,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Our hyperparameters
        "ours": {
            "latent_dim": 8,
            "num_hidden_predictor": 1,
            "hidden_dim_predictor": 250,
            "num_hidden_encoder": 1,
            "hidden_dim_encoder": 100,
            "num_samples_train": 100,
            "num_samples_predict": 200,
            "num_samples_acquire": 200,
            "ib_beta": 0.0001,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 256,
            "patience": 5,
        },
    },
    #############################################################
    #############################################################
    ######################       Cube         ###################
    #############################################################
    #############################################################
    "cube": {
        # RL hyperparameters
        "opportunistic": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "gamma": 0.25,
            "num_samples_predict": 50,
            "num_episodes": 20000,
            "lr": 0.001,
            "batchsize": 128,
            "print_every": 500,
            "eval_every": 100,
        },
        # DIME hyperparameters
        "dime": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
            "eps_scale": 0.4,
        },
        # GDFS hyperparameters
        "gdfs": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 2,
            "temp_scale": 0.1,
        },
        # Fixed MLP hyperparameters
        "fixed_mlp": {
            "hidden_dim": 200,
            "num_hidden": 1,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
        },
        # VAE Hyperparameters
        "vae": {
            "latent_dim": 10,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 200,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 100,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Eddi hyperparameters
        "eddi": {
            "c_dim": 100,
            "latent_dim": 50,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 200,
            "epochs": 400,
            "lr": 0.001,
            "batchsize": 256,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Our hyperparameters
        "ours": {
            "latent_dim": 4,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 250,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 150,
            "num_samples_train": 100,
            "num_samples_predict": 200,
            "num_samples_acquire": 200,
            "ib_beta": 0.005,
            "epochs": 120,
            "lr": 0.0003,
            "batchsize": 128,
            "patience": 5,
        },
    },
    #############################################################
    #############################################################
    #####################     Miniboone       ###################
    #############################################################
    #############################################################
    "miniboone": {
        # RL hyperparameters
        "opportunistic": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "gamma": 0.25,
            "num_samples_predict": 50,
            "num_episodes": 20000,
            "lr": 0.0001,
            "batchsize": 128,
            "print_every": 500,
            "eval_every": 100,
        },
        # DIME hyperparameters
        "dime": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
            "eps_scale": 0.4,
        },
        # GDFS hyperparameters
        "gdfs": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": False,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 2,
            "temp_scale": 1.0,
        },
        # Fixed MLP hyperparameters
        "fixed_mlp": {
            "hidden_dim": 250,
            "num_hidden": 3,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 256,
            "patience": 10,
        },
        # VAE Hyperparameters
        "vae": {
            "latent_dim": 20,
            "num_hidden_decoder": 3,
            "hidden_dim_decoder": 250,
            "num_hidden_encoder": 3,
            "hidden_dim_encoder": 250,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 100,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 512,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Eddi hyperparameters
        "eddi": {
            "c_dim": 60,
            "latent_dim": 60,
            "num_hidden_decoder": 1,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 3,
            "hidden_dim_encoder": 200,
            "epochs": 400,
            "lr": 0.001,
            "batchsize": 512,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Our hyperparameters
        "ours": {
            "latent_dim": 4,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 180,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 40,
            "num_samples_train": 100,
            "num_samples_predict": 200,
            "num_samples_acquire": 200,
            "ib_beta": 0.0008,
            "epochs": 120,
            "lr": 0.0005,
            "batchsize": 128,
            "patience": 8,
        },
    },
    #############################################################
    #############################################################
    #####################        Bank         ###################
    #############################################################
    #############################################################
    "bank": {
        # RL hyperparameters
        "opportunistic": {
            "hidden_dim": 100,
            "num_hidden": 2,
            "gamma": 0.5,
            "num_samples_predict": 50,
            "num_episodes": 20000,
            "lr": 0.001,
            "batchsize": 256,
            "print_every": 500,
            "eval_every": 100,
        },
        # DIME hyperparameters
        "dime": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
            "eps_scale": 0.4,
        },
        # GDFS hyperparameters
        "gdfs": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": False,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 2,
            "temp_scale": 0.1,
        },
        # Fixed MLP hyperparameters
        "fixed_mlp": {
            "hidden_dim": 250,
            "num_hidden": 3,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 256,
            "patience": 10,
        },
        # VAE Hyperparameters
        "vae": {
            "latent_dim": 20,
            "num_hidden_decoder": 3,
            "hidden_dim_decoder": 250,
            "num_hidden_encoder": 3,
            "hidden_dim_encoder": 250,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 100,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 512,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Eddi hyperparameters
        "eddi": {
            "c_dim": 100,
            "latent_dim": 40,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 75,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 75,
            "epochs": 400,
            "lr": 0.001,
            "batchsize": 256,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Our hyperparameters
        "ours": {
            "latent_dim": 6,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 150,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 50,
            "num_samples_train": 100,
            "num_samples_predict": 200,
            "num_samples_acquire": 200,
            "ib_beta": 0.0005,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
        },
    },
    #############################################################
    #############################################################
    #################     California Housing      ###############
    #############################################################
    #############################################################
    "california_housing": {
        # RL hyperparameters
        "opportunistic": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "gamma": 0.25,
            "num_samples_predict": 50,
            "num_episodes": 20000,
            "lr": 0.001,
            "batchsize": 128,
            "print_every": 500,
            "eval_every": 100,
        },
        # DIME hyperparameters
        "dime": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
            "eps_scale": 0.1,
        },
        # GDFS hyperparameters
        "gdfs": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 2,
            "temp_scale": 1.0,
        },
        # Fixed MLP hyperparameters
        "fixed_mlp": {
            "hidden_dim": 250,
            "num_hidden": 3,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 256,
            "patience": 10,
        },
        # VAE Hyperparameters
        "vae": {
            "latent_dim": 50,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 150,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 150,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 200,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Eddi hyperparameters
        "eddi": {
            "c_dim": 250,
            "latent_dim": 250,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 100,
            "num_hidden_encoder": 3,
            "hidden_dim_encoder": 100,
            "epochs": 400,
            "lr": 0.001,
            "batchsize": 128,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Our hyperparameters
        "ours": {
            "latent_dim": 4,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 180,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 40,
            "num_samples_train": 100,
            "num_samples_predict": 200,
            "num_samples_acquire": 200,
            "ib_beta": 0.0008,
            "epochs": 120,
            "lr": 0.0005,
            "batchsize": 128,
            "patience": 8,
        },
    },
    #############################################################
    #############################################################
    #####################       MNIST        ####################
    #############################################################
    #############################################################
    "mnist": {
        # RL hyperparameters
        "opportunistic": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "gamma": 0.25,
            "num_samples_predict": 50,
            "num_episodes": 20000,
            "lr": 0.001,
            "batchsize": 128,
            "print_every": 500,
            "eval_every": 100,
        },
        # DIME hyperparameters
        "dime": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
            "eps_scale": 0.1,
        },
        # GDFS hyperparameters
        "gdfs": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 2,
            "temp_scale": 0.1,
        },
        # Fixed MLP hyperparameters
        "fixed_mlp": {
            "hidden_dim": 250,
            "num_hidden": 3,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 256,
            "patience": 10,
        },
        # VAE Hyperparameters
        "vae": {
            "latent_dim": 50,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 150,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 150,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 200,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Eddi hyperparameters
        "eddi": {
            "c_dim": 250,
            "latent_dim": 250,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 100,
            "num_hidden_encoder": 3,
            "hidden_dim_encoder": 100,
            "epochs": 400,
            "lr": 0.001,
            "batchsize": 128,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Our hyperparameters
        "ours": {
            "latent_dim": 6,
            "num_hidden_predictor": 3,
            "hidden_dim_predictor": 250,
            "num_hidden_encoder": 3,
            "hidden_dim_encoder": 100,
            "num_samples_train": 100,
            "num_samples_predict": 200,
            "num_samples_acquire": 200,
            "ib_beta": 0.001,
            "epochs": 120,
            "lr": 0.0005,
            "batchsize": 256,
            "patience": 5,
        },
    },
    #############################################################
    #############################################################
    #################       Fashion MNIST        ################
    #############################################################
    #############################################################
    "fashion_mnist": {
        # RL hyperparameters
        "opportunistic": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "gamma": 0.25,
            "num_samples_predict": 50,
            "num_episodes": 20000,
            "lr": 0.001,
            "batchsize": 128,
            "print_every": 500,
            "eval_every": 100,
        },
        # DIME hyperparameters
        "dime": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
            "eps_scale": 0.4,
        },
        # GDFS hyperparameters
        "gdfs": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 512,
            "patience": 2,
            "temp_scale": 0.1,
        },
        # Fixed MLP hyperparameters
        "fixed_mlp": {
            "hidden_dim": 250,
            "num_hidden": 3,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 256,
            "patience": 10,
        },
        # VAE Hyperparameters
        "vae": {
            "latent_dim": 50,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 1,
            "hidden_dim_encoder": 150,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 200,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 256,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Eddi hyperparameters
        "eddi": {
            "c_dim": 100,
            "latent_dim": 50,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 200,
            "epochs": 400,
            "lr": 0.001,
            "batchsize": 256,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Our hyperparameters
        "ours": {
            "latent_dim": 6,
            "num_hidden_predictor": 3,
            "hidden_dim_predictor": 250,
            "num_hidden_encoder": 3,
            "hidden_dim_encoder": 100,
            "num_samples_train": 100,
            "num_samples_predict": 200,
            "num_samples_acquire": 200,
            "ib_beta": 0.001,
            "epochs": 120,
            "lr": 0.0005,
            "batchsize": 256,
            "patience": 5,
        },
    },
    #############################################################
    #############################################################
    ####################       Metabric        ##################
    #############################################################
    #############################################################
    "metabric": {
        # RL hyperparameters
        "opportunistic": {
            "hidden_dim": 100,
            "num_hidden": 1,
            "gamma": 0.25,
            "num_samples_predict": 50,
            "num_episodes": 20000,
            "lr": 0.001,
            "batchsize": 128,
            "print_every": 500,
            "eval_every": 100,
        },
        # DIME hyperparameters
        "dime": {
            "hidden_dim": 100,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 2,
            "eps_scale": 0.2,
        },
        # GDFS hyperparameters
        "gdfs": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": False,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 2,
            "temp_scale": 1.0,
        },
        # Fixed MLP hyperparameters
        "fixed_mlp": {
            "hidden_dim": 300,
            "num_hidden": 2,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 256,
            "patience": 5,
        },
        # VAE Hyperparameters
        "vae": {
            "latent_dim": 30,
            "num_hidden_decoder": 1,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 1,
            "hidden_dim_encoder": 200,
            "num_hidden_predictor": 1,
            "hidden_dim_predictor": 200,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Eddi hyperparameters
        "eddi": {
            "c_dim": 100,
            "latent_dim": 50,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 200,
            "epochs": 400,
            "lr": 0.001,
            "batchsize": 256,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Our hyperparameters
        "ours": {
            "latent_dim": 4,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 250,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 150,
            "num_samples_train": 100,
            "num_samples_predict": 200,
            "num_samples_acquire": 200,
            "ib_beta": 0.005,
            "epochs": 120,
            "lr": 0.0003,
            "batchsize": 128,
            "patience": 5,
        },
    },
    #############################################################
    #############################################################
    ######################       TCGA        ####################
    #############################################################
    #############################################################
    "tcga": {
        # RL hyperparameters
        "opportunistic": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "gamma": 0.25,
            "num_samples_predict": 50,
            "num_episodes": 20000,
            "lr": 0.0001,
            "batchsize": 128,
            "print_every": 500,
            "eval_every": 100,
        },
        # DIME hyperparameters
        "dime": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": True,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
            "eps_scale": 0.4,
        },
        # GDFS hyperparameters
        "gdfs": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "share_parameters": False,
            "pretrain_epochs": 80,
            "pretrain_lr": 0.001,
            "epochs": 15,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 2,
            "temp_scale": 1.0,
        },
        # Fixed MLP hyperparameters
        "fixed_mlp": {
            "hidden_dim": 200,
            "num_hidden": 2,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
        },
        # VAE Hyperparameters
        "vae": {
            "latent_dim": 30,
            "num_hidden_decoder": 1,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 1,
            "hidden_dim_encoder": 200,
            "num_hidden_predictor": 1,
            "hidden_dim_predictor": 200,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Eddi hyperparameters
        "eddi": {
            "c_dim": 100,
            "latent_dim": 50,
            "num_hidden_decoder": 2,
            "hidden_dim_decoder": 200,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 200,
            "epochs": 400,
            "lr": 0.001,
            "batchsize": 256,
            "num_acquisition_samples": 50,
            "sig": 0.2,
            "patience": 5,
        },
        # Our hyperparameters
        "ours": {
            "latent_dim": 6,
            "num_hidden_predictor": 2,
            "hidden_dim_predictor": 150,
            "num_hidden_encoder": 2,
            "hidden_dim_encoder": 50,
            "num_samples_train": 100,
            "num_samples_predict": 200,
            "num_samples_acquire": 200,
            "ib_beta": 0.0005,
            "epochs": 120,
            "lr": 0.001,
            "batchsize": 128,
            "patience": 5,
        },
    },
}
