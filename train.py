# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import argparse
import os
import gym
import time
import json
import dmc2gym

import sys
sys.path.append('/home/dell/deep_bisim4control/CARLA_0.9.6/PythonAPI/carla/')
sys.path.append('/home/dell/deep_bisim4control/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')
sys.path.append('/home/dell/deep_bisim4control/CARLA_0.9.6/PythonAPI/carla/agents/navigation/')

import utils
from logger import Logger
from video import VideoRecorder

from agent.baseline_agent import BaselineAgent
from agent.bisim_agent import BisimAgent
from agent.deepmdp_agent import DeepMDPAgent
from agents.navigation.carla_env import CarlaEnv


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='carla')
    parser.add_argument('--task_name', default='highway')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--frame_stack', default=1, type=int)
    parser.add_argument('--resource_files', type=str)
    parser.add_argument('--eval_resource_files', type=str)
    parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
    parser.add_argument('--total_frames', default=10000, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=60000, type=int)
    # train
    parser.add_argument('--agent', default='bisim', type=str, choices=['baseline', 'bisim', 'deepmdp'])
    parser.add_argument('--init_steps', default=100, type=int)
    parser.add_argument('--num_train_steps', default=100000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
    parser.add_argument('--load_encoder', default=None, type=str)
    # eval
    parser.add_argument('--eval_freq', default=10, type=int)  # TODO: master had 10000
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='PixelLstmEncoderCarla096', type=str, choices=['pixel', 'pixelCarla096', 'pixelCarla098','PixelLstmEncoderCarla096', 'identity'])
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    parser.add_argument('--encoder_stride', default=2, type=int)
    parser.add_argument('--decoder_type', default='identity', type=str, choices=['pixel', 'identity', 'contrastive', 'reward', 'inverse', 'reconstruction'])
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.01, type=float)
    parser.add_argument('--alpha_lr', default=1e-3, type=float)
    parser.add_argument('--alpha_beta', default=0.9, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='./save', type=str)
    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--transition_model_type', default='probabilistic', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--port', default=2000, type=int)
    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step, max_hist_len, device=None, embed_viz_dir=None, do_carla_metrics=True):
    # carla metrics:
    reason_each_episode_ended = []
    distance_driven_each_episode = []
    total_dist_driven_this_episode = 0

    crash_intensity = 0.
    steer = 0.
    brake = 0.
    count = 0
    print("Start evaluate !!!!!!!!")

    # embedding visualization
    obses = []
    values = []
    embeddings = []

    for i in range(num_episodes):
        # carla metrics:
        dist_driven_this_episode = 0.
        obs = env.reset()

        # reset history buffer
        o_buff = np.zeros([max_hist_len, *env.observation_space.shape])
        a_buff = np.zeros([max_hist_len, *env.action_space.shape])
        o_buff[0, :] = obs
        o_buff_len = 0

        step_episode = 0

        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            # sample action for data collection
            if step_episode < 5:
                action = env.action_space.sample()
            else:
                with utils.eval_mode(agent):
                    action = agent.select_action(obs, o_buff, a_buff, o_buff_len)
                    # print("Testing! I am thinking how to drive !")

            if embed_viz_dir:
                obses.append(obs)
                with torch.no_grad():
                    values.append(min(agent.critic(torch.Tensor(obs).to(device).unsqueeze(0),
                                                   torch.Tensor(action).to(device).unsqueeze(0))).item())
                    embeddings.append(
                        agent.critic.encoder(torch.Tensor(obs).unsqueeze(0).to(device)).cpu().detach().numpy())

            obs, reward, done, info = env.step(action)
            step_episode += 1

            # Add short history
            if o_buff_len == max_hist_len:
                o_buff[:max_hist_len - 1] = o_buff[1:]
                a_buff[:max_hist_len - 1] = a_buff[1:]
                o_buff[max_hist_len - 1] = list(obs)
                a_buff[max_hist_len - 1] = list(action)
            else:
                o_buff[o_buff_len + 1 - 1] = list(obs)
                a_buff[o_buff_len + 1 - 1] = list(action)
                o_buff_len += 1

            # metrics:
            if do_carla_metrics:
                dist_driven_this_episode += info['distance']
                crash_intensity += info['crash_intensity']
                steer += abs(info['steer'])
                brake += info['brake']
                count += 1

            video.record(env)
            episode_reward += reward

        # metrics:
        if do_carla_metrics:
            reason_each_episode_ended.append(info['reason_episode_ended'])
            distance_driven_each_episode.append(dist_driven_this_episode)
            total_dist_driven_this_episode += dist_driven_this_episode      # the total distance in evaluate episode

    video.save('%d.mp4' % step)
    L.log('eval/episode_reward', episode_reward, step)
    L.log('eval/average_distance_episode', total_dist_driven_this_episode/num_episodes, step)

    if embed_viz_dir:
        dataset = {'obs': obses, 'values': values, 'embeddings': embeddings}
        torch.save(dataset, os.path.join(embed_viz_dir, 'train_dataset_{}.pt'.format(step)))

    L.dump(step)

    if do_carla_metrics:
        print('METRICS--------------------------')
        print("reason_each_episode_ended: {}".format(reason_each_episode_ended))
        print("distance_driven_each_episode: {}".format(distance_driven_each_episode))
        print('crash_intensity: {}'.format(crash_intensity / num_episodes))
        print('steer: {}'.format(steer / count))
        print('brake: {}'.format(brake / count))
        print('---------------------------------')


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'baseline':
        agent = BaselineAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    elif args.agent == 'bisim':
        agent = BisimAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            bisim_coef=args.bisim_coef
        )
    elif args.agent == 'deepmdp':
        agent = DeepMDPAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            encoder_stride=args.encoder_stride,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )

    if args.load_encoder:
        model_dict = agent.actor.encoder.state_dict()
        encoder_dict = torch.load(args.load_encoder) 
        encoder_dict = {k[8:]: v for k, v in encoder_dict.items() if 'encoder.' in k}  # hack to remove encoder. string
        agent.actor.encoder.load_state_dict(encoder_dict)
        agent.critic.encoder.load_state_dict(encoder_dict)

    return agent


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    if args.domain_name == 'carla':
        env = CarlaEnv(
            render_display=args.render,  # for local debugging only
            display_text=args.render,  # for local debugging only
            changing_weather_speed=0.1,  # [0, +inf)
            rl_image_size=args.image_size,
            max_episode_steps=1000,
            frame_skip=args.action_repeat,
            is_other_cars=True,
            port=args.port
        )
        # TODO: implement env.seed(args.seed) ?

        eval_env = env
    else:
        env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.resource_files,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat
        )
        env.seed(args.seed)

        eval_env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.eval_resource_files,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat
        )

    # stack several consecutive frames together
    if args.encoder_type.startswith('pixel'):
        env = utils.FrameStack(env, k=args.frame_stack)
        eval_env = utils.FrameStack(eval_env, k=args.frame_stack)

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # device = torch.device('cpu')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)
    # added by 51
    max_hist_len = 5

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_train_steps):
        if done:
            if args.decoder_type == 'inverse':
                for i in range(1, args.k):  # fill k_obs with 0s if episode is done
                    replay_buffer.k_obses[replay_buffer.idx - i] = 0
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            if episode % args.eval_freq == 0:
                L.log('eval/episode', episode, step)
                # added by 51
                evaluate(eval_env, agent, video, args.num_eval_episodes, L, step, max_hist_len)
                if args.save_model:
                    agent.save(model_dir, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0

            # added by 51 : reset history buffer
            o_buff = np.zeros([max_hist_len, *env.observation_space.shape])
            a_buff = np.zeros([max_hist_len, *env.action_space.shape])
            o_buff[0, :] = obs
            o_buff_len = 0
            reset_flag = 0

            L.log('train/episode', episode, step)

        # sample action for data collection
        if (step < args.init_steps) or (reset_flag < 5):
            action = env.action_space.sample()
            reset_flag += 1
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs, o_buff, a_buff, o_buff_len)
                # print("I am thinking how to drive !")
                # print("action = ", action)

        # run training update
        if step >= args.init_steps:
            if step % 10 == 0:
                num_updates = 50 if step == args.init_steps else 1
                for n in range(num_updates):
                    agent.update(replay_buffer, L, step, max_hist_len)
                    print("I am updating ! ", n)

        curr_reward = reward
        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward

        replay_buffer.add(obs, action, curr_reward, reward, next_obs, done_bool)
        np.copyto(replay_buffer.k_obses[replay_buffer.idx - args.k], next_obs)

        # Add short history
        if o_buff_len == max_hist_len:
            o_buff[:max_hist_len - 1] = o_buff[1:]
            a_buff[:max_hist_len - 1] = a_buff[1:]
            o_buff[max_hist_len - 1] = list(obs)
            a_buff[max_hist_len - 1] = list(action)
        else:
            o_buff[o_buff_len + 1 - 1] = list(obs)
            a_buff[o_buff_len + 1 - 1] = list(action)
            o_buff_len += 1

        obs = next_obs
        episode_step += 1


def collect_data(env, agent, num_rollouts, path_length, checkpoint_path):
    rollouts = []
    for i in range(num_rollouts):
        obses = []
        acs = []
        rews = []
        observation = env.reset()
        for j in range(path_length):
            action = agent.sample_action(observation)
            next_observation, reward, done, _ = env.step(action)
            obses.append(observation)
            acs.append(action)
            rews.append(reward)
            observation = next_observation
        obses.append(next_observation)
        rollouts.append((obses, acs, rews))

    from scipy.io import savemat

    savemat(
        os.path.join(checkpoint_path, "dynamics-data.mat"),
        {
            "trajs": np.array([path[0] for path in rollouts]),
            "acs": np.array([path[1] for path in rollouts]),
            "rews": np.array([path[2] for path in rollouts])
        }
    )


if __name__ == '__main__':
    main()
