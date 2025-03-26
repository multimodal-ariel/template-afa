from __future__ import annotations

import copy
import logging
import os
import pprint
from time import time
from typing import Optional, Type

import numpy as np
import torch as th
import tqdm.auto as tqdm
from jafalib.environment import Env, multirange
from jafalib.model import DFSNet
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def sample(
    q_val: th.Tensor,
    available: th.Tensor,
    exist: th.Tensor,
    eps: Optional[th.Tensor] = None,
) -> tuple[th.Tensor, np.ndarray]:
    """Sample action from q(.|s) specified by q_val.


    Parameters
    ----------
    q_val : 2-D FloatTensor (batch_size x n_actions)
        Q-value
    available : ByteTensor
        Indicator for avaiable action
    exist : ByteTennsor
        Indicator for existing features in the original data.
        To check whether initial state or not
    eps : FloatTensor, optional
        eps for eps-greedy exploration policy


    Returns
    -------
    action : 1-D IntTensor
        chosen action
    max_q_val : 1-D ndarray
        maximum q-value
    """
    assert len(q_val.size()) == 2
    assert available.size() == exist.size()
    N, n_actions = q_val.size()
    assert available.size()[1] + 1 == n_actions
    if eps is not None:
        assert len(eps.size()) == 1 and (eps.size()[0] == 1 or eps.size()[0] == N)
    exploration_prob = th.ones(N, n_actions, out=q_val.new())

    # At least one feature has to be found
    # In the initial state, stop action is not avaiable
    initial = (1 - th.eq(available, exist)).long()
    ind = th.nonzero(initial.sum(dim=1) == 0).squeeze(-1)
    if len(ind) > 0:
        ind = th.stack([ind, ind.new(len(ind)).fill_(q_val.size()[-1] - 1)], dim=-1)
        q_val[ind[:, 0], ind[:, 1]] = -np.inf
        exploration_prob[ind[:, 0], ind[:, 1]] = 0

    # Only available features
    if not available.all():
        ind = th.nonzero(1 - available)
        q_val[ind[:, 0], ind[:, 1]] = -np.inf
        exploration_prob[ind[:, 0], ind[:, 1]] = 0

    max_q_val, action = q_val.max(dim=1)
    noise = q_val.new(N).uniform_()
    if eps is not None:
        noise = q_val.new(N).uniform_()
        exploration = noise < eps
        if exploration.any():
            while True:
                random_action = th.multinomial(
                    exploration_prob[th.nonzero(exploration)[:, 0]],
                    1,
                    replacement=True,
                ).squeeze()
                action[exploration] = random_action
                nonterminal = action < (n_actions - 1)
                bool_ts = th.nonzero(exploration.view(-1) & nonterminal.view(-1)).view(
                    -1
                )
                if (
                    len(bool_ts) == 0
                    or (exploration_prob[bool_ts, action[bool_ts].view(-1)] > 0).all()
                ):
                    assert (
                        len(bool_ts) == 0
                        or available[bool_ts, action[bool_ts].view(-1)].all()
                    )
                    break
    nonterminal = th.nonzero(action < (n_actions - 1)).view(-1)
    if len(nonterminal):
        assert available[nonterminal, action[nonterminal].view(-1)].all()

    return action, max_q_val.detach().cpu().numpy()


def binary_cross_entropy_with_logits(
    input: th.Tensor,
    target: th.Tensor,
    pos_weight: int = 1,
    size_average: bool = True,
    reduce: bool = True,
) -> th.Tensor:
    """calc binary cross entropy with logits


    Parameters
    ----------
    input : 1-D FloatTensor
        logits
    target : 1-D LongTensor
        0 or 1 indicator for binary class
    pos_weight : int, optional
        Unbalanced data handling by using weighted cross entropy loss
    size_average : bool
        If it is false, this func returns 1-D vector


    Returns
    -------
    loss : 0-D or 1-D FloatTensor
        binary cross entropy (averaged if size_average==True
    """
    if not (target.size() == input.size()):
        raise ValueError(
            "Target size ({}) must be the same as input size ({})".format(
                target.size(), input.size()
            )
        )

    max_val = (-input).clamp(min=0)
    l = 1 + (pos_weight - 1) * target
    loss = (
        input
        - input * target
        + l * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
    )

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class StepRunner(object):
    """
    running model for one step,
    geting target q value from target network,
    classification,
    getting final reward from classifier,
    saving and loading model parameter,
    pretraining,
    training with history data
    """

    def __init__(self, model: DFSNet, args):
        self.model = model
        n_features = model.n_features
        n_actions = model.n_actions
        n_classes = model.n_classes
        old_model = copy.deepcopy(self.model)
        if args.batch_size == 0:
            batch_size = args.nsteps * args.n_envs
        else:
            batch_size = args.batch_size

        def step(
            inputs, acquired: th.Tensor, exist, eps=None, acquired_aux=None
        ) -> tuple[th.Tensor, np.ndarray, np.ndarray]:
            length = th.sum(acquired.long(), 1)
            if n_features + 1 == n_actions:
                available = exist * (1 - acquired)  # acquired_aux
            else:
                available = exist * (1 - acquired_aux)
            inputs = inputs.to(args.device)
            length = length.to(args.device)
            available = available.to(args.device)
            exist = exist.to(args.device)
            eps = eps.to(args.device) if eps is not None else None
            p_y_logit, qval, weight = self.model(inputs, length)
            ind, q_a = sample(qval, available, exist, eps=eps)
            return ind, weight, q_a

        def target_q_val(
            inputs, acquired, exist, actions=None, acquired_aux=None
        ) -> th.Tensor | np.ndarray:
            """Target Q-value (bootstrapping)
            for double dqn (choose action with current param and get old val)

            Returns
            -------
            target q-value : FloatTensor
                If actions is not given maximum q-value
            """
            N = inputs.size()[0]
            length = th.sum(acquired.long(), 1)
            available = (
                exist * (1 - acquired)
                if n_features + 1 == n_actions
                else exist * (1 - acquired_aux)
            )
            inputs = inputs.to(args.device)
            length = length.to(args.device)
            available = available.to(args.device)
            exist = exist.to(args.device)
            p_y_logit, qval, weight = old_model(inputs, length)

            if actions is not None:
                # double DQN
                return qval[th.arange(N, out=actions.new()), actions]
            else:
                # vanila Q-learning (maximum Q-value)
                ind, q_a = sample(qval, available, exist)
                return q_a  # return Tensor

        def update_target():
            old_model.load_state_dict(self.model.state_dict())

        def classify(inputs, acquired) -> th.Tensor:
            self.model.eval()
            """Get classifier output (logits) """
            length = th.sum(acquired.long(), 1)
            inputs = inputs.to(args.device)
            length = length.to(args.device)
            p_y_logit = self.model(inputs, length)[0]
            self.model.train()
            return p_y_logit

        def calc_final_reward(p_y_logit, labels) -> th.Tensor:
            if n_classes > 2:
                crss_ent = th.nn.functional.cross_entropy(
                    p_y_logit, labels, reduce=False
                )
            else:
                crss_ent = binary_cross_entropy_with_logits(
                    p_y_logit.contiguous().view(-1), labels.float(), reduce=False
                )
            loglikelihood = -crss_ent
            return loglikelihood

        def save(save_path):
            th.save(model.state_dict(), save_path)

        def load(save_path):
            model.load_state_dict(th.load(save_path))

        def get_clf_loss(p_y_logit, labels, pos_weight=1):
            if n_classes > 2:
                return th.nn.functional.cross_entropy(p_y_logit, labels.long())
            else:
                return binary_cross_entropy_with_logits(
                    p_y_logit.view(-1), labels.float(), pos_weight=pos_weight
                )

        def pretrain_step(
            clf_optimizer, full_obs, full_labels, full_length, pos_weight=1
        ):
            full_obs = full_obs.to(args.device)
            full_length = full_length.to(args.device)
            full_labels = full_labels.to(args.device)
            # clf loss
            p_y_logit, _, weight = self.model(full_obs, full_length)
            clf_loss = get_clf_loss(p_y_logit, full_labels, pos_weight=pos_weight)
            clf_optimizer.zero_grad()
            clf_loss.backward()  # retain_graph=True)
            clf_optimizer.step()

        def train_step(
            obs,
            acquired,
            exist,
            returns,
            actions,
            labels,
            acquired_aux,
            full_obs,
            full_labels,
            full_length,
            iter,
            pos_weight=1,
            policy_optimizer=None,
            clf_optimizer=None,
        ):
            """Get running history and adjust model parameter


            Parameters
            ----------
            policy_optimizer : th.optim
            clf_optimizer : th.optim
            obs : FloatTensor
                masked information
            acquired : ByteTensor
                indicator whether acquired or not
            exist : ByteTensor
                indicator whether unmissing feature or not
            returns : FloatTensor
                estimated returns (cumulative reward)
            actions : LongTensor

            labels: LongTensor

            acquired_aux : ByteTensor
                On some dataset a group of features are acquired at once
                This is to handle when action space is not directly mapped to
                feature indicies
            """
            obs = obs.to(args.device)
            labels = labels.to(args.device)
            actions = actions.to(args.device)
            returns = returns.to(args.device)
            if n_actions != n_features + 1:
                acquired_aux = acquired_aux.to(args.device)
            acquired = acquired.to(args.device)
            exist = exist.to(args.device)
            full_obs = full_obs.to(args.device)
            full_length = full_length.to(args.device)
            full_labels = full_labels.to(args.device)

            # acquired_aux
            if n_actions == n_features + 1:
                available = (1 - acquired) * exist
            else:
                available = (1 - acquired_aux) * exist
            length = th.sum(acquired.long(), 1)

            sampler = BatchSampler(
                SubsetRandomSampler(range(obs.size()[0])), batch_size, drop_last=False
            )
            for indices in sampler:
                indices = th.LongTensor(indices)
                indices = indices.to(args.device)
                obs_ = obs[indices]
                labels_ = labels[indices]
                actions_ = actions[indices]
                returns_ = returns[indices]
                acquired_ = acquired[indices]
                if n_actions != n_features + 1:
                    acquired_aux_ = acquired_aux[indices]
                exist_ = exist[indices]
                if n_actions == n_features + 1:
                    available_ = (1 - acquired_) * exist_  # acquired_aux
                else:
                    available_ = (1 - acquired_aux_) * exist_
                length_ = th.sum(acquired_.long(), 1)
                # policy update
                p_y_logit, qval, weight = self.model(obs_, length_)
                q_a = inputs = qval[
                    th.arange(len(indices), out=actions.new()), actions_
                ]
                targets = returns_
                if args.done_action_train:
                    inputs = th.cat((inputs, qval[:, n_actions - 1]), dim=0)
                    final_reward = calc_final_reward(p_y_logit, labels_)
                    targets = th.cat((targets, final_reward.detach()), dim=0)
                policy_loss = th.nn.functional.smooth_l1_loss(inputs, targets.detach())
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                if clf_optimizer is not None:
                    # clf loss
                    if args.complete:
                        obs_ = th.cat((obs_, full_obs), dim=0)
                        length_ = th.cat((length_, full_length), dim=0)
                        labels_ = th.cat((labels_, full_labels), dim=0)
                    p_y_logit, _, weight = self.model(obs_, length_)
                    clf_loss = get_clf_loss(p_y_logit, labels_, pos_weight=pos_weight)
                    clf_optimizer.zero_grad()
                    clf_loss.backward()
                    clf_optimizer.step()
                else:
                    clf_loss = None

        update_target()

        self.step = step
        self.save = save
        self.load = load
        self.classify = classify
        self.pretrain_step = pretrain_step
        self.target_q_val = target_q_val
        self.update_target = update_target
        self.train_step = train_step


def run_n_steps(
    step_runner: StepRunner,
    env: Env,
    n_steps: int,
    eps: Optional[th.Tensor] = None,
    mode="double",
):
    """Generate history for training"""
    n_features = env.n_features
    n_actions = env.n_actions
    n_envs = env.n_envs
    need_aux = n_actions != n_features + 1
    mb_obs, mb_exist, mb_acquired, mb_labels = [], [], [], []
    mb_actions, mb_dones, mb_rewards = [], [], []
    mb_acquired_aux = [] if need_aux else None

    obs = env.var_obs
    acquired: th.Tensor | None = None
    exist: th.Tensor | None = None
    acquired_aux: th.Tensor | None = None
    for step in range(n_steps):
        acquired = env.var_acquired
        exist = env.var_exist
        labels = env.var_labels
        acquired_aux = env.var_acquired_aux if need_aux else None

        mb_obs.append(obs.clone())
        mb_acquired.append(acquired.clone())
        if need_aux and mb_acquired_aux is not None:
            mb_acquired_aux.append(acquired_aux)
        mb_exist.append(exist.clone())
        mb_labels.append(labels.clone())
        # run model

        actions, _, q_a = step_runner.step(
            obs, acquired, exist, eps, acquired_aux=acquired_aux
        )
        mb_actions.append(actions.cpu())
        # interact with environment
        obs, rewards, dones = env.step(actions)
        mb_rewards.append(rewards.clone())
        mb_dones.append(dones.clone())
    assert acquired is not None
    assert exist is not None
    # s_(t+1)
    actions = (
        step_runner.step(obs, acquired, exist, acquired_aux=acquired_aux)[0]
        if mode == "double"
        else None
    )
    q_val = step_runner.target_q_val(
        obs, acquired, exist, actions, acquired_aux=acquired_aux
    )
    # n_steps x batch_size

    # FIXME oh.............
    mb_obs = th.stack(mb_obs).view(-1, *env.input_size)
    mb_exist = th.stack(mb_exist).view(-1, n_actions - 1)
    mb_acquired = th.stack(mb_acquired).view(-1, n_features)
    if need_aux:
        mb_acquired_aux = th.stack(mb_acquired_aux).view(-1, n_actions - 1)
    mb_labels = th.stack(mb_labels).view(-1)

    mb_actions = th.stack(mb_actions)
    mb_rewards = th.stack(mb_rewards)
    mb_dones = th.stack(mb_dones)

    # n_steps x n_envs
    mb_returns = []
    R = q_val.cpu() if isinstance(q_val, th.Tensor) else q_val  # FIXME
    for i in range(n_steps)[::-1]:
        R = R * (1.0 - mb_dones[i].float())
        R = R + mb_rewards[i]
        mb_returns.append(R.clone())
    mb_returns = th.stack(mb_returns[::-1])  # step x n_env
    mb_actions = mb_actions.view(-1)
    mb_returns = mb_returns.view(-1)

    return (
        mb_obs,
        mb_acquired,
        mb_exist,
        mb_returns,
        mb_actions,
        mb_labels,
        mb_acquired_aux,
    )


def test(step_runner: StepRunner, env: Env, args, iter=0):
    args_str = pprint.pformat(vars(args))
    n_classes = env.n_classes
    n_envs = env.n_envs
    n_features = env.n_features
    n_actions = env.n_actions
    n_data = env.n_data
    need_aux = n_actions != n_features + 1

    obs = env.var_obs
    offset = 0
    inputs = np.zeros((n_data, n_features))
    acquired = np.zeros((n_data, n_features))
    exist = np.zeros((n_data, n_features))
    if need_aux:
        acquired_aux = np.zeros((n_data, n_actions - 1))
    labels = np.zeros(n_data)
    correct = np.zeros(n_data)
    probs = np.zeros(n_data)
    returns = np.zeros(n_data)
    weights = np.zeros((n_data, n_features))
    order = np.zeros((n_data, n_features))
    q_a = np.zeros((n_data, n_actions))
    if n_classes == 2:
        sigmoid = np.zeros(n_data)
    offset = 0

    while offset < n_data:
        acquired_ = env.var_acquired
        acquired_aux = env.var_acquired_aux if need_aux else None
        exist_ = env.var_exist
        labels_ = env.var_labels
        # run model
        actions, weights_, q_a_ = step_runner.step(
            obs, acquired_, exist_, acquired_aux=acquired_aux
        )
        obs, done_records = env.test_step(actions, q_a_)
        # record
        if done_records[0] is not None and len(done_records[0]):
            tmp = [done_records[i].shape[0] for i in range(7)]
            assert tmp[0] == tmp[1] == tmp[2] == tmp[3] == tmp[4] == tmp[5]
            n_terminal = done_records[0].shape[0]
            from_ = offset
            to_ = offset + n_terminal
            inputs[from_:to_] = done_records[0]
            acquired[from_:to_] = done_records[1]
            labels[from_:to_] = done_records[2]
            correct[from_:to_] = done_records[3]
            probs[from_:to_] = done_records[4]
            returns[from_:to_] = done_records[5]
            if weights_ is not None:
                weights[from_:to_] = weights_[done_records[6]]
            q_a[from_:to_] = env.q_a[done_records[6]]
            try:
                order[from_:to_] = done_records[7]
            except:
                pass
            if n_classes == 2:
                sigmoid[from_:to_] = done_records[8]
            offset = to_
    assert offset == n_data
    lgr = logging.Logger("jafalib.main.test")
    lgr.info(args_str)
    lgr.info(
        f"accuracy {np.mean(correct)}",
    )
    if n_classes == 2:
        auc = roc_auc_score(labels, sigmoid)
        lgr.info(f"auc {auc}")
    lgr.info(f"n_acquired(mean) {np.mean(np.sum(acquired, 1))}")
    lgr.info(f"n_acquired(min) {np.amin(np.sum(acquired, 1))}")
    lgr.info(f"n_acquired(max) {np.amax(np.sum(acquired, 1))}")
    lgr.info(f"n_acquired(med) {np.median(np.sum(acquired, 1))}")
    lgr.info(f"picked detail {list(enumerate(np.sum(acquired, 0).astype(int)))}")
    if weights_ is not None:
        lgr.info(f"weight {weights.sum(axis=0) / acquired.sum(axis=0)}")
    lgr.info(f"returns(mean) {np.mean(returns)}")
    lgr.info(f"returns(min) {np.amin(returns)}")
    lgr.info(f"returns(max) {np.amax(returns)}")
    lgr.info(f"returns(med) {np.median(returns)}")

    # print(args_str)
    # print("accuracy", np.mean(correct))
    # if n_classes == 2:
    #     auc = roc_auc_score(labels, sigmoid)
    #     print("auc", auc)
    # print("n_acquired(mean)", np.mean(np.sum(acquired, 1)))
    # print("n_acquired(min)", np.amin(np.sum(acquired, 1)))
    # print("n_acquired(max)", np.amax(np.sum(acquired, 1)))
    # print("n_acquired(med)", np.median(np.sum(acquired, 1)))
    # print("picked detail", list(enumerate(np.sum(acquired, 0).astype(int))))
    # if weights_ is not None:
    #     print("weight", weights.sum(axis=0) / acquired.sum(axis=0))
    # print("returns(mean)", np.mean(returns))
    # print("returns(min)", np.amin(returns))
    # print("returns(max)", np.amax(returns))
    # print("returns(med)", np.median(returns))

    if n_classes > 2:
        return (
            inputs,
            correct,
            acquired,
            returns,
            weights,
            probs,
            labels,
            order,
            q_a,
            exist,
        )
    return (
        inputs,
        correct,
        acquired,
        returns,
        weights,
        probs,
        labels,
        order,
        q_a,
        exist,
        auc,
    )


def learn(
    step_runner: StepRunner,
    args,
    env: Env,
    valenv: Optional[Env] = None,
    nsteps: int = 5,
    total_steps: int = int(80e6),
    lr: float = 7e-4,
    scheduler: str = "linear",
    optim: Type[th.optim.Optimizer] = Adam,
):
    # TODO lr_scheduler
    n_features, n_classes = env.n_features, env.n_classes

    if args.batch_size == 0:
        batch_size = args.nsteps * args.n_envs
    else:
        batch_size = args.batch_size
    mult = 1 if args.dropout else 0
    params = [
        {
            "params": step_runner.model.clf_weight_params,
            "weight_decay": mult * (1 - args.p) / batch_size,
        },
        {
            "params": step_runner.model.clf_bias_params,
            "weight_decay": mult * 1 / batch_size,
        },
        {"params": step_runner.model.encoder.parameters()},
    ]
    clf_optimizer = optim(params, lr=lr)
    # policy_optimizer = optim(step_runner.model.policy.parameters(),
    #        lr=lr, weight_decay=0)
    policy_optimizer = optim(
        list(step_runner.model.policy.parameters())
        + list(step_runner.model.encoder.parameters()),
        lr=lr,
        weight_decay=0,
    )  # type:ignore
    n = total_steps // (nsteps * env.n_envs)
    update = 0
    if "test" not in args.message:
        if scheduler == "linear":
            decay = args.decay_rate * (args.eps_start - args.eps_end) / (n)
        else:
            decay = 0.999
        eps = args.eps_start
        max_score = 0
        ###################
        # Pretraining start
        ###################
        valdata = valenv.data.features
        valexist = (
            np.ones_like(valdata) if valenv.data.exist is None else valenv.data.exist
        )
        vallength = th.from_numpy(np.sum(valexist.astype(int), axis=1)).to(args.device)
        valinput = np.zeros((len(valdata), n_features, n_features + 1)).astype(
            np.float32
        )
        x, y = np.where(valexist)
        y_ = multirange(vallength.cpu())
        valinput[x, y_, 0] = valdata[x, y]
        valinput[x, y_, y + 1] = 1
        valinput = th.from_numpy(valinput).to(args.device)
        valtarget = valenv.data.labels
        max_val_score = 0
        pretrain_start = time()
        pbar = tqdm.trange(
            args.pretrain, desc="pretrain", leave=True, dynamic_ncols=True
        )
        for pre_i in pbar:
            env.reset(first=False)
            if args.pretrain_sample == "full":
                step_runner.pretrain_step(
                    clf_optimizer,
                    *env.get_current_batch_with_all_features(),
                    pos_weight=env.pos_weight,
                )
            elif args.pretrain_sample == "random":
                full_obs, full_labels, full_length = (
                    env.get_current_batch_with_random_features()
                )
                step_runner.pretrain_step(
                    clf_optimizer,
                    full_obs,
                    full_labels,
                    full_length,
                    pos_weight=env.pos_weight,
                )
            else:  # 'both'
                full_obs, full_labels, full_length = (
                    env.get_current_batch_with_all_features()
                )
                full_obs_, full_labels_, full_length_ = (
                    env.get_current_batch_with_random_features()
                )
                full_obs = th.cat((full_obs, full_obs_), 0)
                full_labels = th.cat((full_labels, full_labels_), 0)
                full_length = th.cat((full_length, full_length_), 0)
                step_runner.pretrain_step(
                    clf_optimizer,
                    full_obs,
                    full_labels,
                    full_length,
                    pos_weight=env.pos_weight,
                )

            if (pre_i + 1) % 10 == 0:
                step_runner.model.eval()
                vallogit, _, weight = step_runner.model(valinput, vallength)
                vallogit = vallogit.detach().cpu().numpy()
                val_score = (
                    roc_auc_score(valtarget, expit(vallogit))
                    if env.n_classes == 2
                    else np.mean(valtarget == vallogit.argmax(axis=1))
                )
                val_score = (
                    val_score.item() if isinstance(val_score, np.ndarray) else val_score
                )
                # print(pre_i+1, val_score)
                if val_score > max_val_score:
                    # print(pre_i + 1, val_score, "SAVE")
                    step_runner.save(
                        os.path.join(args.save_path, "pretrained_best.model")
                    )
                    max_val_score = val_score
                pbar.set_postfix({"score": val_score, "max_score": max_val_score})
                step_runner.model.train()
        pbar.close()
        pretrain_time = time() - pretrain_start
        step_runner.load(os.path.join(args.save_path, "pretrained_best.model"))
        env.reset()
        ###################
        # Pretraining end
        ###################
        pbar = tqdm.trange(
            1,
            n + 1,
            desc="train policy",
            leave=True,
            dynamic_ncols=True,
        )
        for update in pbar:
            eps = (
                max(args.eps_end, eps - decay) if scheduler == "linear" else decay * eps
            )
            eps_ = (
                th.linspace(0.1 * eps, eps, env.n_envs)
                if env.n_envs > 2
                else th.FloatTensor([eps])
            )
            run_history = run_n_steps(
                step_runner, env, nsteps, eps=eps_, mode=args.mode
            )
            step_runner.train_step(
                *run_history,
                *env.get_current_batch_with_all_features(),
                update,
                pos_weight=env.pos_weight,
                policy_optimizer=policy_optimizer,
                clf_optimizer=clf_optimizer,
            )
            if update % args.target_update_freq:
                step_runner.update_target()
            if update % 100 == 0 or update == n:
                step_runner.model.eval()
                pbar.set_postfix({"eps": eps})
                # print()
                # print(update, "/", n)
                # print("Current eps", eps)
                if valenv is not None:
                    valenv.reset()
                    if n_classes == 2:
                        score = test(step_runner, valenv, args, iter=update)[-1]
                    else:
                        correct = test(step_runner, valenv, args, iter=update)[1]
                        score = np.mean(correct)
                    if score >= max_score and update > (n / 3):  # premature..
                        max_score = score
                        main_train_opt = time()
                        step_runner.save(
                            os.path.join(args.save_path, "trained_best.model")
                        )
                step_runner.model.train()
        pbar.close()
