# %%
import _classifiers
import mydatasets.aaco
import mymodels.classifiers
import numpy as np
import torch as th


# %%
def eval_masks(
    X,
    B,
    classifiers: mymodels.classifiers.SubsetFeatureClassifier,
    Y=None,
    bsize=16,
    matched_B=False,
    classifier_filters=None,
):
    """
    Evaluate an ensemble of classifiers on tuples of inputs and masks.
    Args:
      X: N x d data matrix
      B: M x d mask matrix
      Y: N array of labels. If Y is given then the negative log likelihoods
        is returned else predictions are returned.
      If matched_B is True then N==M and instances are only evaluated on
        respective masks; else (default), all instances are evaluated on each
        mask in B.
      classifier_filter: N x nests matrix of indicators of which classifiers to
        use for instances
    Returns: N x M nlls if Y is given else N x M x nclasses predictions (when
      matched_B=False).
    """
    N = X.shape[0]
    M = B.shape[0]

    if Y is not None:
        if matched_B:
            out = np.zeros((N,), np.float32)
        else:
            out = np.zeros((N, M), np.float32)
    else:
        if matched_B:
            out = np.zeros((N, classifiers.n_labels), np.float32)
        else:
            out = np.zeros((N, M, classifiers.n_labels), np.float32)
    for bi in range(0, N, bsize):
        Xbatch = X[bi : bi + bsize]
        Ybatch = Y[bi : bi + bsize]
        if matched_B:
            Bbatch = B[bi : bi + bsize]
        else:
            Bbatch = np.tile(B, (Xbatch.shape[0], 1))
            Xbatch = np.repeat(Xbatch, M, 0)

        # Xbatch = np.concatenate((Xbatch * Bbatch, Bbatch), 1)
        preds: np.ndarray = classifiers.predict_proba(
            th.as_tensor(Xbatch * Bbatch, dtype=th.float32),
            th.as_tensor(Bbatch, dtype=th.float32),
        ).numpy()
        # preds = np.stack([est.predict_proba(Xbatch) for est in classifiers], -1)
        # if classifier_filters is None:
        #     preds = np.mean(preds, -1)
        # else:
        #     split_batch = classifier_filters[bi : bi + bsize]
        #     split_batch = np.repeat(split_batch, M, 0)
        #     nests = np.sum(split_batch, -1, keepdims=True)
        #     # print(preds.shape)
        #     # print(split_batch.shape)
        #     # print(nests.shape)
        #     preds = np.sum(preds * split_batch[:, None, :], -1) / nests
        if not matched_B:
            preds = preds.reshape(-1, M, classifiers.n_labels)
        if Y is None:
            out[bi : bi + bsize, ...] = preds
        else:
            out[bi : bi + bsize, ...] = np.stack(
                [-np.log(preds[i, ..., y]) for i, y in enumerate(Ybatch)], 0
            )
    return out


def get_mask_losses(
    Xtrn, Ytrn, B, classifiers, featcost, bsize=256, classifier_filters=None
):
    # TODO: use filters?
    mask_costs = featcost * np.sum(B, 1)

    bsize = 256
    Xcosts = []
    for si in range(0, Xtrn.shape[0], bsize):
        print(si)
        if classifier_filters is not None:
            nlls = eval_masks(
                Xtrn[si : si + bsize],
                B,
                classifiers,
                Y=Ytrn[si : si + bsize],
                classifier_filters=classifier_filters[si : si + bsize],
            )
        else:
            nlls = eval_masks(
                Xtrn[si : si + bsize], B, classifiers, Y=Ytrn[si : si + bsize]
            )
        Xcosts.append(nlls + mask_costs)

    return np.concatenate(Xcosts, 0)


def tf_arbitrary_knn(Xtrn, Ytrn, Xtst, indices, k=1, Xtrnl2=None, toss_first=False):
    """
    Args:
      Xtrn: N x d Train Instances
      Ytrn: N x nclass Train Labels (e.g., one-hot)
      Xtst: M x d Query Instances
      indices: list of features to use
      k: number of neighbors
      Xtrnl2: N x 1 vector of squared norms of Xtrn instances
      toss_first: flag to throwout the first neighbor
                  (if querying within training set)
    """

    # Xtrnfeats = np.take_along_axis(Xtrn, indices, axis=1)
    # Xtstfeats = np.take_along_axis(Xtst, indices, axis=1)
    Xtrnfeats = Xtrn[:, indices]
    Xtstfeats = Xtst[:, indices]
    if Xtrnl2 is None:
        Xtrnl2 = np.sum(Xtrnfeats**2, axis=1, keepdims=True)
    d2 = (
        Xtrnl2
        - 2.0 * np.matmul(Xtrnfeats, Xtstfeats.T)
        + np.transpose(np.sum(Xtstfeats**2, axis=1, keepdims=True))
    )
    d2_sorti = np.argsort(d2, axis=0)
    ntmpl: int = Ytrn.shape[1]
    n: int = d2_sorti.shape[1]
    d2_sorti = th.as_tensor(d2_sorti)[:, None, :].expand(-1, ntmpl, -1).numpy()
    ys = th.as_tensor(Ytrn)[:, :, None].expand(-1, -1, n).numpy()
    Y_neighbors = np.mean(
        np.take_along_axis(
            ys,
            d2_sorti[(1 if toss_first else 0) : (k + 1 if toss_first else k), :, :],
            axis=0,
        ),
        axis=0,
    ).T
    return Y_neighbors


def eval_rollout(
    Xval,
    Yval,
    startdim,
    featcost,
    Xtrn,
    Ytrn,
    tempNLLtrn,
    B_temp,
    ao_k=5,
    bank=None,
    classifiers=None,
):
    d = Xval.shape[1]
    eyed = np.eye(d, dtype=np.float32)
    eyedplusone = np.eye(d + 1, dtype=np.float32)

    accu_ao = []
    bs_ao = []
    # ao_k = 5
    for i in range(Xval.shape[0]):
        print("\n\n\n{}".format(i))
        # i = np.random.randint(Xval.shape[0])
        for s in range(d + 1):
            if s == 0:
                # if in the first step, draw action from null_distribution
                # b_curr = np.float32(np.random.multinomial(1, p_null))
                b_curr = eyed[startdim, None, :]  # Deterministic start
                print(b_curr)
                continue  # move to next step
            elif s < d:
                xo = Xval[i, None, :] * b_curr

                nllroll = tf_arbitrary_knn(
                    Xtrn, tempNLLtrn, xo, np.flatnonzero(b_curr[0]), k=ao_k
                )

                mask_needed = np.maximum(
                    B_temp - b_curr, 0.0
                )  # fine if we have additional feats
                mask_costs = featcost * np.sum(mask_needed, 1)
                classifier_costs = nllroll + mask_costs
                best_clss = np.argmin(classifier_costs)
                # print('>>{}'.format(np.sort(classifier_costs)))
                # print('>>{}'.format(mask_needed[best_clss, :]))
                a_pred = np.concatenate(
                    [mask_needed[best_clss, :], [0.0]]
                )  # TODO: tie break with softmax combination?
                # a_pred[:-1] = a_pred[:-1] - 1e8*b_curr[0]  # Don't acquire what's already acquired
                a_pred[:-1] = a_pred[:-1] * (1.0 - b_curr[0])
                if np.sum(a_pred[:-1]) == 0.0:
                    a_pred[-1] = 1.0
                else:
                    a_pred[:-1] = a_pred[:-1] / np.sum(a_pred[:-1])
                    a_pred[:-1] = a_pred[:-1] * (1.0 - a_pred[-1])
                action = eyedplusone[:, np.argmax(a_pred)]  # TODO: case with ties?
                # a_pred = le.inverse_transform(
                #   policy.predict_proba(np.concatenate((xo, b_curr), -1))[0, :])
                # action = eyedplusone[:, np.argmax(a_pred)]
                if action[-1] == 0.0:  # not predicting
                    b_curr = b_curr + action[:-1]
                    print(b_curr)
                    continue  # move to next step
            # make prediction
            print(np.flatnonzero(b_curr[0]))
            if bank is not None:
                est = bank.get_estimator(b_curr[0])
                y_pred = est.predict_proba(Xval[i : i + 1, b_curr[0] > 0])
            elif classifiers is not None:
                y_pred = np.mean(
                    np.stack(
                        [
                            est.predict_proba(np.concatenate((xo, b_curr), 1))
                            for est in classifiers
                        ],
                        -1,
                    ),
                    -1,
                )
            else:
                est = XGBClassifier(n_estimators=40)
                est.fit(Xtrn[:, b_curr[0] > 0], Ytrn)
                y_pred = est.predict_proba(Xval[i : i + 1, b_curr[0] > 0])

            print(y_pred, np.argmax(y_pred[0, :]), int(Yval[i]))
            accu_ao.append(np.argmax(y_pred[0, :]) == int(Yval[i]))
            print("correct!") if accu_ao[-1] else print("wrong!")
            bs_ao.append(b_curr)
            print(np.mean(accu_ao))
            print(np.mean([np.sum(b) for b in bs_ao]))
            break
    return np.mean(accu_ao), np.mean([np.sum(b) for b in bs_ao])


# %%
data_name: str = "big5_C_cls"
_tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data(data_name, to_normalize=False)
_tdata_shuffle_idxs = th.randperm(len(_tdata))
tdata = _tdata[_tdata_shuffle_idxs[: len(_tdata) // 2]]
extdata = _tdata[_tdata_shuffle_idxs[len(_tdata) // 2 :]]
n_covs: int = _tdata["xs"].shape[1]
n_labels: int = len(th.unique(_tdata["ys"]))

# %%
tclassifier = _classifiers.SubsetFeatureConcatXGBClassifier(
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
    xgb_kwargs={"n_estimators": 40},
    fraction_training_data_per_split=1.0,
    n_splits=64,
    n_tmpl_per_instance=4,
)

vclassifier = mymodels.classifiers.SubsetFeatureXGBClassifier(
    xs_train=extdata["xs"].numpy(),
    ys_train=extdata["ys"].numpy(),
    xgbc_kwargs={"n_estimators": 40},
)

# %%
init_fidx: int = 35
tmpls: th.Tensor = th.load("tmpls.pt")

# %%
tclassifier.fit_(tmpls)

# %%
tcosts: np.ndarray = get_mask_losses(
    tdata["xs"].numpy(), tdata["ys"].numpy(), tmpls.numpy(), tclassifier, 0.0, bsize=256
)

# %%


# %%
