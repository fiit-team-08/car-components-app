# https://www.youtube.com/watch?v=cO5g5qLrLSo&t=610s
# https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN
# https://github.com/jmpf2018/ShipAI
# https://towardsdatascience.com/openai-gym-from-scratch-619e39af121f
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from ML.RL.car_env_v0.env_v2 import CarEnv
import pickle
# uncomment this line if you don§'t want to use cuda
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# get the environment and extract the number of actions
env = CarEnv()
nb_actions = env.action_space.n

# build a very simple model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# configure and compile the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
               nb_steps_warmup=10000, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Define 'test' for testing an existing network weights or 'train' to train
# a new one!
mode = 'train'

if mode == 'train':
    filename = '400kit_rn4_maior2_mem20k_20acleme_target1000_epsgr1'
    hist = dqn.fit(env, nb_steps=300000, visualize=False, verbose=1)
    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # after training is done, we save the final weights
    dqn.save_weights('dqn_{}_weights.h5f'.format(filename), overwrite=True)
    # evaluate our algorithm for 10 episodes
    dqn.test(env, nb_episodes=10, visualize=True)

if mode == 'test':
    filename = '400kit_rn4_maior2_mem20k_20acleme_target1000_epsgr1'
    dqn.load_weights('h5f_files/dqn_{}_weights.h5f'.format(filename))
    dqn.test(env, nb_episodes=10, visualize=True)
