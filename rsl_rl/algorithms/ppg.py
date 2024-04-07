#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple

from rsl_rl.modules import ActorCriticPpg
from rsl_rl.storage import RolloutStorage




AuxMemory = namedtuple('Memory', ['obs', 'critic_obs', 'returns', 'target_values'])

class ExperienceDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))

def create_shuffled_dataloader(data, batch_size):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size = batch_size, shuffle = True)


class PPG:
    actor_critic: ActorCriticPpg

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        beta_clone=1.0,
        num_policy_updates_per_aux: int = 32,  # N_pi
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO/PPG components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO/PPG parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # PPG components
        self.aux_memories = []
        
        # PPG parameters
        self.update_cnt = 0
        self.num_policy_updates_per_aux = num_policy_updates_per_aux
        self.epochs_aux = 6
        self.beta_clone = beta_clone


    # This method is called in the training loop. CANNOT change the method signature(interface).
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    # Record the transition in 'self.storage'
    # This method is called in the training loop. CANNOT change the method signature(interface).
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    # This method is called in the training loop. CANNOT change the method signature(interface).
    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)


    def value_loss(self, values, returns, target_values):
        if self.use_clipped_value_loss:
            value_clipped = target_values + (values - target_values).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (values - returns).pow(2)
            value_losses_clipped = (value_clipped - returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns - values).pow(2).mean()
        
        return value_loss


    def update_network_(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()


    # Aux phase update
    def update_aux(self):
        # gather states and target values into one tensor
        obs = []
        critic_obs = []
        returns = []
        target_values = []
        for ob, critic_ob, return_, target_value in self.aux_memories:
            obs.append(ob)
            critic_obs.append(critic_ob)
            returns.append(return_)
            target_values.append(target_value)

        obs = torch.cat(obs)
        critic_obs = torch.cat(critic_obs)
        returns = torch.cat(returns)
        target_values = torch.cat(target_values)

        # get old action predictions for minimizing kl divergence and clipping respectively
        self.actor_critic(obs)
        old_mu = self.actor_critic.action_mean
        old_sigma = self.actor_critic.action_std

        # prepared dataloader for auxiliary phase training
        dl = create_shuffled_dataloader([obs, critic_obs, old_mu, old_sigma, returns, target_values], self.minibatch_size)

        # the proposed auxiliary phase training
        # where the value is distilled into the policy network, while making sure the policy network does not change the action predictions (kl div loss)
        for epoch in range(self.epochs_aux):
            for obs, critic_obs, old_mu, old_sigma, returns, target_values in dl:
                
                ## Optimize policy function
                policy_values = self.actor_critic.actor_evaluate(obs)
                aux_loss = self.value_loss(policy_values, returns, target_values)

                self.actor_critic.act(obs)
                mu, sigma = self.actor_critic.action_mean, self.actor_critic.action_std
                kl = torch.sum(
                    torch.log(sigma / old_sigma + 1.0e-5)
                    + (torch.square(old_sigma) + torch.square(old_mu - mu))
                        / (2.0 * torch.square(sigma))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)
                
                joint_loss = aux_loss + self.beta_clone * kl_mean
                self.update_network_(joint_loss)
                
                ## Optimize value function, the same as in the original PPO and the policy phase of PPG
                # Value function loss
                values = self.actor_critic.evaluate(critic_obs)
                value_loss = self.value_loss(values, returns, target_values)
                
                self.update_network_(value_loss)

        # clear auxiliary memory
        self.aux_memories = []


    # This method is called in the training loop. CANNOT change the method signature(interface).
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            # value_loss = self.value_loss(value_batch, returns_batch, target_values_batch)
            
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            # store state and target values to auxiliary memory buffer for later training
            self.aux_memories.append(AuxMemory(obs_batch, critic_obs_batch, returns_batch, target_values_batch))

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()
        
        # Aux phase update
        self.update_cnt += 1
        if self.update_cnt % self.num_policy_updates_per_aux == 0:
            self.update_aux()
        

        return mean_value_loss, mean_surrogate_loss
