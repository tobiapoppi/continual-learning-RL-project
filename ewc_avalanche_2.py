"""
    Example of replicating a small-scale (single gpu) version of the experiments
    presented in EWC original paper https://arxiv.org/abs/1612.00796.
    The experiment interleaves learning of multiple atari games with a single
    network (using a 'cumulative' action space) with an added penalty given by
    the Fisher importance matrix, which discourages updates to the parameters
    which are most important for a previous task. 
"""
import torch, json, gym, time
import numpy as np
from avalanche_rl.training.strategies.buffers import ReplayMemory
from avalanche_rl.training.plugins.rl_plugins import RLEvaluationPlugin
from avalanche_rl.training.strategies.dqn import DQNStrategy, default_dqn_logger
from avalanche_rl.training.strategies.env_wrappers \
    import ReducedActionSpaceWrapper
from avalanche_rl.benchmarks.rl_benchmark_generators \
    import atari_benchmark_generator
from avalanche_rl.training.plugins.ewc import EWCRL
from avalanche_rl.logging import TensorboardLogger
from avalanche_rl.models.dqn import EWCConvDeepQN
from avalanche_rl.training.plugins.rl_plugins import RLStrategyPlugin
from avalanche_rl.training.strategies.rl_base_strategy import Timestep
from avalanche_rl.evaluation.metrics.reward import GenericFloatMetric
from torch.optim import Adam
from gym.wrappers import FrameStack, AtariPreprocessing

if __name__ == "__main__":
    #device = torch.device('cuda:0')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # let's simplify things a bit for the agent:
    # both games can be played with 3 (or 2 without considering NOOP) 
    # actions but the action space is unecessarily big due to action keys
    # ordering; we then reduce the action space
    # to 3 actions and re-map LEFT-RIGHT actions so that we skip FIRE.
    # Actions are re-mapped before the step() method is called.  
    action_space = 3

    def action_wrapper_class(env):
        return ReducedActionSpaceWrapper(
            env, action_space_dim=action_space, action_mapping={1: 2, 2: 3})

    n_envs = 1
    # frameskipping is done in wrapper
    scenario = atari_benchmark_generator(
        #['BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4'],
        ['PongNoFrameskip-v4'],
        n_parallel_envs=n_envs, frame_stacking=True,
        normalize_observations=True, terminal_on_life_loss=True,
        n_experiences=3, 
        extra_wrappers=[action_wrapper_class],
        #eval_envs=['BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4'])
        eval_envs=['PongNoFrameskip-v4'])

    # let's instatiate an external replay memory
    memory_size = 10000
    memory = ReplayMemory(size=memory_size, n_envs=n_envs)
    ewc_plugin = EWCRL(400., memory, mode='separate',
                       start_ewc_after_experience=1)

    # log to tensorboard
    # tb_logger = TensorboardLogger("/tmp/tb_data")

    # keep track of the loss
    ewc_penalty_metric = GenericFloatMetric(
        'loss', 'Ewc Loss', reset_value=0., emit_on=['after_backward'],
        update_on=['before_backward'])

    evaluator = RLEvaluationPlugin(
        *default_dqn_logger.metrics, ewc_penalty_metric,
        loggers=default_dqn_logger.loggers)

    # here we'll have task-specific biases & gains per layer
    # (2 since we're learning 2 games)
    model = EWCConvDeepQN(4, (84, 84), 
                          action_space,
                          #4, 
                          n_tasks=1, bias=True)
    print('Model', model)
    optimizer = Adam(model.parameters(), lr=1e-4)

    # a custom plugin to show some functionalities: halve inital epsilon
    # (for eps-greedy action-selection)
    # every two training experiences, so that more exploit is done
    class HalveEps(RLStrategyPlugin):

        def __init__(self):
            super().__init__()

        def after_training_exp(self, strategy: DQNStrategy, **kwargs):
            if strategy.training_exp_counter % 2 == 1:
                strategy._init_eps = strategy._init_eps / 2

        def after_training_iteration(self, strategy, **kwargs):
            pass

    # adapted hyperparams from:
    # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
    # we run 6 experiences, alternating between the 2 games:
    # the first two are longer (1e5 steps) the rest are shorter (3e4 steps)
    strategy = DQNStrategy(
        model, optimizer,
        per_experience_steps=[Timestep(int(1e4))],
        #per_experience_steps=[Timestep(int(2e3)),
        #                      Timestep(int(2e3))],
        batch_size=128, exploration_fraction=.15,
        final_epsilon=.01, max_steps_per_rollout=4,
        plugins=[ewc_plugin, HalveEps()],
        # external replay memory is automatically filled with initial size and
        # reset on new experience
        initial_replay_memory=memory, replay_memory_init_size=4000,
        double_dqn=True,
        target_net_update_interval=1000, eval_every=int(5e3),
        eval_episodes=6, evaluator=evaluator, device=device)

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Env ", experience.env)

        strategy.train(experience, [scenario.eval_stream])
        # print('Training completed')
        # save model and optimizer
        torch.save(
            {'model': model.state_dict(),
             'optim': optimizer.state_dict()},
            'pong.pt')



    def test_model(model, env, episodes=100):
        scores = []
        for _ in range(episodes):
            time.sleep(0.5)
            state = env.reset()
            done = False
            score = 0
            while not done:
                # Assuming state shape is [210, 160, 4] after preprocessing and stacking
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to [1, 4, 210, 160]
                
                # No need to check the length of state; it should be consistent after wrapping
                action = model(state_tensor, 0)[0].argmax().item()

                next_state, reward, done, _ = env.step(action)
                score += 1
                state = next_state
            scores.append(score)
        return np.mean(scores)

    # Wrap the environment with preprocessing and frame stacking
    #env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
    #env = AtariPreprocessing(env)
    #env = FrameStack(env, num_stack=4)

    #test_model(model, env)
    


