from noge.envs import OnlineGraphEnv
from noge.envs import TargetMeasEnvWrapper


def make_env(max_episode_steps,
             nn_feat,
             max_nodes,
             max_edges,
             data_generator,
             seed,
             reward_type='path_length',
             temporal_coeffs=None,
             meas_coeffs=None,
             goal_space=None,
             sample_goals=None):

    # env config
    env = OnlineGraphEnv(max_episode_steps=max_episode_steps,
                         reward_type=reward_type,
                         max_nodes=max_nodes,
                         nn_feat=nn_feat,
                         max_edges=max_edges,
                         data_generator=data_generator)

    if temporal_coeffs is not None:
        env = TargetMeasEnvWrapper(env=env,
                                   meas_coeffs=meas_coeffs,
                                   temporal_coeffs=temporal_coeffs,
                                   goal_space=goal_space,
                                   sample_goals=sample_goals)

    env.seed(seed)
    return env


def make_memory(memory_type, features, online=False, **kwargs):
    assert memory_type == 'cat'

    from noge.graph_memories import CategoricalOnlineMemory, CategoricalOfflineMemory

    features = features[memory_type]

    if online:
        mem = CategoricalOnlineMemory(features=features, **kwargs)
    else:
        mem = CategoricalOfflineMemory(features=features, **kwargs)

    return mem
