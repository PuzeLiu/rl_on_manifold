import argparse
import os
import pandas as pd
from tqdm import trange
from mushroom_rl.algorithms.actor_critic import PPO, TRPO, DDPG, TD3, SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.policy import GaussianTorchPolicy, OrnsteinUhlenbeckPolicy, ClippedGaussianPolicy
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.utils.preprocessors import MinMaxPreprocessor
from atacom.environments.planar_air_hockey import AirHockeyPlanarAtacom, \
    AirHockeyHitUnconstrained, AirHockeyDefendUnconstrained
from network import *

def experiment(seed, results_dir, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               quiet, **kwargs):
    mdp = build_env(**kwargs)

    agent, build_params = build_agent(mdp_info=mdp.info, **kwargs)

    logger = Logger(results_dir=results_dir, seed=seed, log_name='exp')

    logger.strong_line()
    logger.info('Experiment Algorithm: ' + type(agent).__name__)
    if hasattr(mdp, "env"):
        logger.info('Environment: ' + type(mdp.env).__name__ + " seed: " + str(seed))
    else:
        logger.info('Environment: ' + type(mdp).__name__ + " seed: " + str(seed))

    # normalization callback
    prepro = MinMaxPreprocessor(mdp_info=mdp.info)

    core = Core(agent, mdp, preprocessors=[prepro])

    eval_params = dict(
        n_episodes=n_episodes_test,
        render=False,
        quiet=quiet
    )

    J, R, E, c_avg, c_max, c_dq_max = compute_metrics(core, eval_params, build_params)
    best_J, best_R, best_E, best_c_avg, best_c_max, best_c_dq_max = J, R, E, c_avg, c_max, c_dq_max

    logger.epoch_info(0, J=J, R=R, E=E, c_avg=c_avg, c_max=c_max, c_dq_max=c_dq_max)
    logger.log_numpy(J=J, R=R, E=E, c_avg=c_avg, c_max=c_max, c_dq_max=c_dq_max)
    logger.log_agent(agent)
    prepro.save(os.path.join(logger.path, "state_normalization" + logger._suffix + ".msh"))

    for it in trange(n_epochs, leave=False, disable=quiet):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=quiet)
        J, R, E, c_avg, c_max, c_dq_max = compute_metrics(core, eval_params, build_params)

        logger.epoch_info(it + 1, J=J, R=R, E=E, c_avg=c_avg, c_max=c_max, c_dq_max=c_dq_max)
        logger.log_numpy(J=J, R=R, E=E, c_avg=c_avg, c_max=c_max, c_dq_max=c_dq_max)

        if J > best_J:
            best_J = J
            best_R = R
            best_E = E
            best_c_avg = c_avg
            best_c_max = c_max
            best_c_dq_max = c_dq_max

            logger.log_agent(agent)
            prepro.save(os.path.join(logger.path, "state_normalization" + logger._suffix + ".msh"))

    logger.info("Best result | J: {}, R: {}, E:{}, c_avg:{}, c_max:{}, c_dq_max{}.".format(best_J, best_R, best_E,
                                                                                           best_c_avg, best_c_max,
                                                                                           best_c_dq_max))
    logger.strong_line()
    best_res = {"best_J": best_J, "best_R": best_R, "best_E": best_E,
                "best_c_avg": best_c_avg, "best_c_max": best_c_max, "best_c_dq_max": best_c_dq_max}
    best_res = pd.DataFrame.from_dict(best_res, orient="index")
    best_res.to_csv(os.path.join(logger.path, "best_result.csv"))


def compute_metrics(core, eval_params, build_params):
    dataset = core.evaluate(**eval_params)
    c_avg, c_max, c_dq_max = 0., 0., 0.
    if hasattr(core.mdp, "get_constraints_logs"):
        c_avg, c_max, c_dq_max = core.mdp.get_constraints_logs()
    J = np.mean(compute_J(dataset, core.mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = None
    if build_params['compute_policy_entropy']:
        if build_params['compute_entropy_with_states']:
            E = core.agent.policy.entropy(parse_dataset(dataset)[0])
        else:
            E = core.agent.policy.entropy()
    return J, R, E, c_avg, c_max, c_dq_max


def build_env(**kwargs):
    env = kwargs['env']
    gamma = kwargs['gamma']
    random_init = kwargs['random_init']
    debug_gui = kwargs['debug_gui']

    iiwa_hit_horizon = kwargs['iiwa_hit_horizon']
    iiwa_hit_time_step = kwargs['iiwa_hit_time_step']
    iiwa_hit_n_intermediate_steps = kwargs['iiwa_hit_n_intermediate_steps']

    if env == 'H':
        mdp = AirHockeyPlanarAtacom(task=env, horizon=120, gamma=gamma, random_init=random_init,
                                         timestep=1 / 240., n_intermediate_steps=4,
                                         debug_gui=debug_gui, Kc=240.)
    elif env == 'D':
        mdp = AirHockeyPlanarAtacom(task=env, horizon=180, gamma=gamma, random_init=random_init,
                                         timestep=1 / 240., n_intermediate_steps=4,
                                         debug_gui=debug_gui, Kc=240.)
    elif env == 'UH':
        mdp = AirHockeyHitUnconstrained(horizon=120, gamma=gamma, random_init=random_init,
                                   timestep=1 / 240., n_intermediate_steps=4,
                                   debug_gui=debug_gui)
    elif env == 'UD':
        mdp = AirHockeyDefendUnconstrained(horizon=180, gamma=gamma, random_init=random_init,
                                            timestep=1 / 240., n_intermediate_steps=4,
                                            debug_gui=debug_gui)
    else:
        raise NotImplementedError
    return mdp


def build_agent(alg, mdp_info, **kwargs):
    if isinstance(kwargs['n_features'], str):
        kwargs['n_features'] = kwargs['n_features'].split(' ')

    alg = alg.upper()
    if alg == 'PPO':
        agent, build_params = build_agent_PPO(mdp_info, **kwargs)
    elif alg == 'TRPO':
        agent, build_params = build_agent_TRPO(mdp_info, **kwargs)
    elif alg == 'DDPG':
        agent, build_params = build_agent_DDPG(mdp_info, **kwargs)
    elif alg == 'TD3':
        agent, build_params = build_agent_TD3(mdp_info, **kwargs)
    elif alg == 'SAC':
        agent, build_params = build_agent_SAC(mdp_info, **kwargs)
    else:
        raise NotImplementedError
    return agent, build_params


def build_agent_PPO(mdp_info, actor_lr, critic_lr, n_features, batch_size, eps_ppo, lam, ent_coeff, use_cuda, **kwargs):
    policy_params = dict(
        std_0=0.5,
        n_features=n_features,
        use_cuda=use_cuda
    )
    policy = GaussianTorchPolicy(PPONetwork,
                                 mdp_info.observation_space.shape,
                                 mdp_info.action_space.shape,
                                 **policy_params)

    critic_params = dict(network=PPONetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': critic_lr}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         batch_size=batch_size,
                         input_shape=mdp_info.observation_space.shape,
                         output_shape=(1,))

    ppo_params = dict(actor_optimizer={'class': optim.Adam,
                                       'params': {'lr': actor_lr}},
                      n_epochs_policy=4,
                      batch_size=batch_size,
                      eps_ppo=eps_ppo,
                      lam=lam,
                      ent_coeff=ent_coeff,
                      critic_params=critic_params)

    build_params = dict(compute_entropy_with_states=False,
                        compute_policy_entropy=True)

    return PPO(mdp_info, policy, **ppo_params), build_params


def build_agent_TRPO(mdp_info, critic_lr, n_features, batch_size, lam, ent_coeff, use_cuda,
                     max_kl, n_epochs_line_search, n_epochs_cg, cg_damping, cg_residual_tol, critic_fit_params,
                     **kwargs):
    policy_params = dict(
        std_0=0.5,
        n_features=n_features,
        use_cuda=use_cuda
    )

    critic_params = dict(
        network=TRPONetwork,
        optimizer={'class': optim.Adam,
                   'params': {'lr': critic_lr}},
        loss=F.mse_loss,
        n_features=n_features,
        batch_size=batch_size,
        input_shape=mdp_info.observation_space.shape,
        output_shape=(1,))

    trpo_params = dict(
        ent_coeff=ent_coeff,
        max_kl=max_kl,
        lam=lam,
        n_epochs_line_search=n_epochs_line_search,
        n_epochs_cg=n_epochs_cg,
        cg_damping=cg_damping,
        cg_residual_tol=cg_residual_tol,
        critic_fit_params=critic_fit_params)

    policy = GaussianTorchPolicy(TRPONetwork,
                                 mdp_info.observation_space.shape,
                                 mdp_info.action_space.shape,
                                 **policy_params)

    build_params = dict(compute_entropy_with_states=False,
                        compute_policy_entropy=True)

    return TRPO(mdp_info, policy, critic_params, **trpo_params), build_params


def build_agent_DDPG(mdp_info, actor_lr, critic_lr, n_features, batch_size,
                     initial_replay_size, max_replay_size, tau, use_cuda, **kwargs):
    policy_params = dict(
        sigma=np.ones(1) * .2,
        theta=0.15,
        dt=1e-2)

    actor_params = dict(
        network=DDPGActorNetwork,
        input_shape=mdp_info.observation_space.shape,
        output_shape=mdp_info.action_space.shape,
        action_scaling=(mdp_info.action_space.high - mdp_info.action_space.low) / 2,
        n_features=n_features,
        use_cuda=use_cuda)

    actor_optimizer = {
        'class': optim.Adam,
        'params': {'lr': actor_lr}}

    critic_params = dict(
        network=DDPGCriticNetwork,
        optimizer={'class': optim.Adam,
                   'params': {'lr': critic_lr}},
        loss=F.mse_loss,
        n_features=n_features,
        batch_size=batch_size,
        input_shape=(mdp_info.observation_space.shape[0] + mdp_info.action_space.shape[0],),
        action_shape=mdp_info.action_space.shape,
        output_shape=(1,),
        action_scaling=(mdp_info.action_space.high - mdp_info.action_space.low) / 2,
        use_cuda=use_cuda)

    alg_params = dict(
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        batch_size=batch_size,
        tau=tau)

    build_params = dict(compute_entropy_with_states=False,
                        compute_policy_entropy=False)

    return DDPG(mdp_info, OrnsteinUhlenbeckPolicy, policy_params, actor_params, actor_optimizer, critic_params,
                **alg_params), build_params


def build_agent_TD3(mdp_info, actor_lr, critic_lr, n_features, batch_size, use_cuda,
                    initial_replay_size, max_replay_size, tau, sigma, **kwargs):
    policy_params = dict(
        sigma=np.eye(mdp_info.action_space.shape[0]) * sigma,
        low=mdp_info.action_space.low,
        high=mdp_info.action_space.high)

    actor_params = dict(
        network=TD3ActorNetwork,
        input_shape=mdp_info.observation_space.shape,
        output_shape=mdp_info.action_space.shape,
        action_scaling=(mdp_info.action_space.high - mdp_info.action_space.low) / 2,
        n_features=n_features,
        use_cuda=use_cuda)

    actor_optimizer = {
        'class': optim.Adam,
        'params': {'lr': actor_lr}}

    critic_params = dict(
        network=TD3CriticNetwork,
        optimizer={'class': optim.Adam,
                   'params': {'lr': critic_lr}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=(mdp_info.observation_space.shape[0] + mdp_info.action_space.shape[0],),
        action_shape=mdp_info.action_space.shape,
        output_shape=(1,),
        action_scaling=(mdp_info.action_space.high - mdp_info.action_space.low) / 2,
        use_cuda=use_cuda)

    alg_params = dict(
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        batch_size=batch_size,
        tau=tau)

    build_params = dict(compute_entropy_with_states=False,
                        compute_policy_entropy=False)

    return TD3(mdp_info, ClippedGaussianPolicy, policy_params, actor_params, actor_optimizer, critic_params,
               **alg_params), build_params


def build_agent_SAC(mdp_info, actor_lr, critic_lr, n_features, batch_size,
                    initial_replay_size, max_replay_size, tau,
                    warmup_transitions, lr_alpha, target_entropy, use_cuda,
                    **kwargs):
    actor_mu_params = dict(network=SACActorNetwork,
                           input_shape=mdp_info.observation_space.shape,
                           output_shape=mdp_info.action_space.shape,
                           n_features=n_features,
                           use_cuda=use_cuda)
    actor_sigma_params = dict(network=SACActorNetwork,
                              input_shape=mdp_info.observation_space.shape,
                              output_shape=mdp_info.action_space.shape,
                              n_features=n_features,
                              use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': actor_lr}}
    critic_params = dict(network=SACCriticNetwork,
                         input_shape=(mdp_info.observation_space.shape[0] + mdp_info.action_space.shape[0],),
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': critic_lr}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    alg_params = dict(initial_replay_size=initial_replay_size,
                      max_replay_size=max_replay_size,
                      batch_size=batch_size,
                      warmup_transitions=warmup_transitions,
                      tau=tau,
                      lr_alpha=lr_alpha,
                      critic_fit_params=None,
                      target_entropy=target_entropy)

    build_params = dict(compute_entropy_with_states=True,
                        compute_policy_entropy=True)

    return SAC(mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params,
               **alg_params), build_params


def default_params():
    defaults = dict(env='H', alg='SAC', seed=1,
                    gamma=0.99, random_init=False, debug_gui=False, quiet=False, use_cuda=False,
                    iiwa_hit_time_step=1/240., iiwa_hit_n_intermediate_steps=4, iiwa_hit_horizon=100,
                    results_dir="../logs/planar_air_hockey")
    training_params = dict(n_epochs=100, n_steps=3000, n_steps_per_fit=600, n_episodes_test=25)

    network_params = dict(actor_lr=3e-4, critic_lr=3e-4, n_features=[64, 64], batch_size=64)

    trpo_ppo_params = dict(lam=0.95, ent_coeff=5e-5)
    ppo_params = dict(eps_ppo=0.1)
    trpo_params = dict(max_kl=1e-2, n_epochs_line_search=10, n_epochs_cg=10, cg_damping=1e-2, cg_residual_tol=1e-10,
                       critic_fit_params=None)

    ddpg_td3_sac_params = dict(initial_replay_size=5000, max_replay_size=200000, tau=1e-3)
    td3_params = dict(sigma=0.25)

    sac_params = dict(warmup_transitions=10000, lr_alpha=3e-4, target_entropy=-6)

    defaults.update(training_params)
    defaults.update(network_params)
    defaults.update(trpo_ppo_params)
    defaults.update(ppo_params)
    defaults.update(trpo_params)
    defaults.update(ddpg_td3_sac_params)
    defaults.update(td3_params)
    defaults.update(sac_params)
    return defaults


def parse_args():
    parser = argparse.ArgumentParser()

    arg_test = parser.add_argument_group('Experiment')
    arg_test.add_argument('--env', choices=['H', 'D', 'UH', 'UD'],
                          help="Environment argument ['H', 'D', 'UH', 'UD']: "
                               "H for Hitting using ATACOM, "
                               "D for Defending using ATACOM."
                               "UH for unconstrained Hitting."
                               "UD for unconstrained Defending.")
    arg_test.add_argument('--alg', choices=['TRPO', 'trpo', 'PPO', 'ppo', 'DDPG', 'ddpg', 'TD3', 'td3', 'SAC', 'sac'])

    arg_test.add_argument('--gamma', type=float)
    arg_test.add_argument('--random-init', action="store_true")
    arg_test.add_argument('--termination-tol', type=float)
    arg_test.add_argument('--debug-gui', action="store_true")
    arg_test.add_argument('--quiet', action="store_true")
    arg_test.add_argument('--use-cuda', action="store_true")

    arg_test.add_argument('--iiwa-hit-horizon', type=int)
    arg_test.add_argument('--iiwa-hit-time-step', type=float)
    arg_test.add_argument('--iiwa-hit-n-intermediate-steps', type=int)

    # training parameter
    arg_test.add_argument('--n-epochs', type=int)
    arg_test.add_argument('--n-steps', type=int)
    arg_test.add_argument('--n-steps-per-fit', type=int)
    arg_test.add_argument('--n-episodes-test', type=int)

    # network parameter
    arg_test.add_argument('--actor-lr', type=float)
    arg_test.add_argument('--critic-lr', type=float)
    arg_test.add_argument('--n-features', nargs='+')
    arg_test.add_argument('--batch-size', type=int)

    # TRPO PPO parameter
    arg_test.add_argument('--lam', type=float)
    arg_test.add_argument('--ent-coeff', type=float)

    # PPO parameters
    arg_test.add_argument('--eps-ppo', type=float)

    # TRPO parameters
    arg_test.add_argument('--max-kl', type=float)
    arg_test.add_argument('--n-epochs-line-search', type=int)
    arg_test.add_argument('--n-epochs-cg', type=int)
    arg_test.add_argument('--cg-damping', type=float)
    arg_test.add_argument('--cg-residual-tol', type=float)

    # DDPG TD3 parameters
    arg_test.add_argument('--initial-replay-size', type=int)
    arg_test.add_argument('--max-replay-size', type=int)
    arg_test.add_argument('--tau', type=float)

    # TD3 parameters
    arg_test.add_argument('--sigma', type=float)

    # SAC parameters
    arg_test.add_argument('--warmup-transitions', type=int)
    arg_test.add_argument('--lr-alpha', type=float)
    arg_test.add_argument('--target-entropy', type=float)

    arg_default = parser.add_argument_group('Default')
    arg_default.add_argument('--seed', type=int)
    arg_default.add_argument('--results-dir', type=str)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args_ = parse_args()
    experiment(**args_)
