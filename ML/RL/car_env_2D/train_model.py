# https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN
import argparse
from collections import deque
from DQNAgent import CarRacingDQNAgent
from utils import process_state_image
from utils import generate_state_frame_stack_from_queue
from car_env import CarEnv
import tensorflow as tf


RENDER = False
STARTING_EPISODE = 1
ENDING_EPISODE = 1000
SKIP_FRAMES = 2
TRAINING_BATCH_SIZE = 64
SAVE_TRAINING_FREQUENCY = 25
UPDATE_TARGET_MODEL_FREQUENCY = 5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training a DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model',
                        help='''Specify the last trained model path if you 
                        want to continue training after it.''')
    parser.add_argument('-s', '--start', type=int,
                        help='The starting episode, default to 1.')
    parser.add_argument('-e', '--end', type=int,
                        help='The ending episode, default to 1000.')
    parser.add_argument('-p', '--epsilon', type=float, default=1.0,
                        help='''The starting epsilon of the agent, 
                             default to 1.0.''')
    args = parser.parse_args()

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(
        physical_devices[0], True
    )

    env = CarEnv()
    agent = CarRacingDQNAgent(epsilon=args.epsilon)
    if args.model:
        agent.load(args.model)
    if args.start:
        STARTING_EPISODE = args.start
    if args.end:
        ENDING_EPISODE = args.end

    for e in range(STARTING_EPISODE, ENDING_EPISODE + 1):
        init_state = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state] * agent.frame_stack_num,
                                        maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False

        while True:
            if RENDER:
                env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(
                state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)

            reward = 0
            for _ in range(SKIP_FRAMES + 1):
                next_state, r, done, info = env.step(action)
                reward += r
                if done:
                    break

            # If continually getting negative reward 10 times after
            # the tolerance steps, terminate this episode
            if time_frame_counter > 100 and reward < 0:
                negative_reward_counter += 1
            else:
                negative_reward_counter = 0

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(
                state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward,
                           next_state_frame_stack, done)

            if done or negative_reward_counter >= 25 or total_reward < 0:
                print(
                    'Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.2}, Epsilon: {:.2}'.format(
                        e, ENDING_EPISODE, time_frame_counter,
                        float(total_reward), float(agent.epsilon)))
                break
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)
            time_frame_counter += 1

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            agent.save('./save/trial_{}.h5'.format(e))

    env.close()
