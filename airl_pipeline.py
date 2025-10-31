"""
AIRL Pipeline using HumanCompatibleAI/imitation library.
Supports vector-based observation environments with pre-trained experts.
"""
import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

import warnings
warnings.filterwarnings("ignore")

# Environment mapping for SEALS pre-trained experts
SEALS_ENV_MAP = {
    "CartPole-v0": "seals:seals/CartPole-v0",
    # "CartPole-v1": "seals:seals/CartPole-v1",  # Map to available expert
    "HalfCheetah-v1": "seals:seals/HalfCheetah-v1",

}

EXPERT_MAP = {
    "CartPole-v0": "seals-CartPole-v0",
    # "CartPole-v1": "seals-CartPole-v1",
    "HalfCheetah-v1": "seals-HalfCheetah-v1",
}


class AIRLPipeline:
    """End-to-end AIRL pipeline for inverse reinforcement learning."""
    
    def __init__(self, env_name: str, seed: int = 42, n_envs: int = 8):
        self.env_name = env_name
        self.seed = seed
        self.n_envs = n_envs
        self.expert = None
        self.rollouts = None
        self.learner = None
        self.airl_trainer = None
        self.venv = None
        
    def setup_environment(self):
        """Create vectorized environment with rollout wrapper."""
        print(f"[1/4] Setting up environment: {self.env_name}")
        
        # Use SEALS environment if available, otherwise standard gym
        env_id = SEALS_ENV_MAP.get(self.env_name, self.env_name)
        
        self.venv = make_vec_env(
            env_id,
            rng=np.random.default_rng(self.seed),
            n_envs=self.n_envs,
            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        )
        print(f"  ✓ Environment ready: {env_id} ({self.n_envs} parallel envs)")
        return self.venv
    
    def load_expert(self):
        """Load pre-trained expert policy from HuggingFace."""
        print(f"[2/4] Loading expert policy from HuggingFace")
        
        if self.venv is None:
            raise ValueError("Environment not set up. Run setup_environment() first.")
        
        # Check if expert available for this environment
        expert_name = EXPERT_MAP.get(self.env_name)
        if expert_name is None:
            raise ValueError(
                f"No pre-trained expert for {self.env_name}. "
                f"Available: {list(EXPERT_MAP.keys())}"
            )
        
        self.expert = load_policy(
            "ppo-huggingface",
            organization="HumanCompatibleAI",
            env_name=expert_name,
            venv=self.venv,
        )
        
        # Evaluate expert
        expert_rewards, _ = evaluate_policy(
            self.expert, self.venv, n_eval_episodes=100, return_episode_rewards=True
        )
        mean_reward = np.mean(expert_rewards)
        std_reward = np.std(expert_rewards)
        
        print(f"  ✓ Expert loaded | Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        return self.expert
    
    def collect_demonstrations(self, n_episodes: int = 60):
        """Collect expert demonstrations."""
        print(f"[3/4] Collecting {n_episodes} expert demonstrations")
        
        if self.expert is None:
            raise ValueError("Expert not loaded. Run load_expert() first.")
        
        self.rollouts = rollout.rollout(
            self.expert,
            self.venv,
            rollout.make_sample_until(min_episodes=n_episodes),
            rng=np.random.default_rng(self.seed),
        )
        
        total_transitions = sum(len(r.obs) for r in self.rollouts)
        print(f"  ✓ Collected {len(self.rollouts)} episodes, {total_transitions} transitions")
        return self.rollouts
    
    def setup_airl(self, learner_config: dict = None, airl_config: dict = None):
        """Initialize AIRL trainer with learner and reward network."""
        print("[4/4] Setting up AIRL trainer")
        
        if self.rollouts is None:
            raise ValueError("No demonstrations. Run collect_demonstrations() first.")
        
        # Default learner config
        if learner_config is None:
            learner_config = {
                "batch_size": 64,
                "ent_coef": 0.0,
                "learning_rate": 5e-4,
                "gamma": 0.95,
                "clip_range": 0.1,
                "vf_coef": 0.1,
                "n_epochs": 5,

            }
        
        # Default AIRL config
        if airl_config is None:
            airl_config = {
                "demo_batch_size": 2048,
                "gen_replay_buffer_capacity": 512,
                "n_disc_updates_per_round": 16,
            }
        
        # Initialize learner (student policy)
        self.learner = PPO(
            env=self.venv,
            policy=MlpPolicy,
            seed=self.seed,
            verbose=0,
            **learner_config,
        )
        
        # Initialize reward network
        reward_net = BasicShapedRewardNet(
            observation_space=self.venv.observation_space,
            action_space=self.venv.action_space,
            normalize_input_layer=RunningNorm,
        )
        
        # Initialize AIRL trainer
        self.airl_trainer = AIRL(
            demonstrations=self.rollouts,
            venv=self.venv,
            gen_algo=self.learner,
            reward_net=reward_net,
           # allow_variable_horizon=True,  # Add this line to prevent extra prints
            **airl_config,
        )
        
        # Calculate minimum timesteps needed
        min_timesteps = (
            airl_config["gen_replay_buffer_capacity"] 
            * airl_config["n_disc_updates_per_round"]
        )
        
        print(f"  ✓ AIRL trainer initialized")
        print(f"  → Minimum training timesteps: {min_timesteps}")
        return self.airl_trainer
    
    def train_airl(self, total_timesteps: int = 100000):
        """Train using AIRL."""
        print(f"\nTraining AIRL ({total_timesteps} steps)")
        print("="*60)
        
        if self.airl_trainer is None:
            raise ValueError("AIRL not set up. Run setup_airl() first.")
        
        # Evaluate before training
        mean_before, std_before = evaluate_policy(
            self.learner, self.venv, n_eval_episodes=20
        )
        print(f"Learner reward before: {mean_before:.2f} ± {std_before:.2f}")

        suppress_prints()
        # Train
        print(f"\nTraining for {total_timesteps} timesteps...")
        self.airl_trainer.train(total_timesteps)
        enable_prints()
        
        # Evaluate after training
        mean_after, std_after = evaluate_policy(
            self.learner, self.venv, n_eval_episodes=20
        )
        
        print(f"\nLearner reward after: {mean_after:.2f} ± {std_after:.2f}")
        print(f"Improvement: {mean_after - mean_before:+.2f}")
        print("="*60)
        
        return {
            "before": (mean_before, std_before),
            "after": (mean_after, std_after),
            "improvement": mean_after - mean_before,
        }
    
    def run_full_pipeline(
        self,
        n_demo_episodes: int = 60,
        airl_timesteps: int = 100000,
        learner_config: dict = None,
        airl_config: dict = None,
    ):
        """Run complete AIRL pipeline end-to-end."""
        self.setup_environment()
        self.load_expert()
        self.collect_demonstrations(n_episodes=n_demo_episodes)
        self.setup_airl(learner_config=learner_config, airl_config=airl_config)
        results = self.train_airl(total_timesteps=airl_timesteps)
        
        print("\n" + "="*60)
        print("Pipeline complete!")
        print("="*60)
        return results

    
def suppress_prints():
    sys.stdout = open(os.devnull, 'w')

def enable_prints():
    sys.stdout = sys.__stdout__ # Restore original stdout

def main():
    parser = argparse.ArgumentParser(description="AIRL Pipeline with Pre-trained Experts")
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v0",
        choices=list(EXPERT_MAP.keys()),
        help="Environment with pre-trained expert",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel envs")
    parser.add_argument(
        "--demo-episodes",
        type=int,
        default=60,
        help="Number of demonstration episodes",
    )
    parser.add_argument(
        "--airl-steps",
        type=int,
        default=100000,
        help="AIRL training timesteps (min: 8192 for default config)",
    )
    parser.add_argument(
        "--demo-batch-size",
        type=int,
        default=2048,
        help="Demonstration batch size",
    )
    parser.add_argument(
        "--gen-buffer-capacity",
        type=int,
        default=512,
        help="Generator replay buffer capacity",
    )
    parser.add_argument(
        "--n-disc-updates",
        type=int,
        default=16,
        help="Number of discriminator updates per round",
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"AIRL Pipeline - {args.env}")
    print("="*60)
    
    # AIRL configuration
    airl_config = {
        "demo_batch_size": args.demo_batch_size,
        "gen_replay_buffer_capacity": args.gen_buffer_capacity,
        "n_disc_updates_per_round": args.n_disc_updates,
    }
    
    # Check minimum timesteps
    min_timesteps = args.gen_buffer_capacity * args.n_disc_updates
    if args.airl_steps < min_timesteps:
        print(f"\n⚠️  WARNING: airl-steps ({args.airl_steps}) < minimum ({min_timesteps})")
        print(f"    Adjusting to minimum: {min_timesteps}\n")
        args.airl_steps = min_timesteps
    
    pipeline = AIRLPipeline(
        env_name=args.env,
        seed=args.seed,
        n_envs=args.n_envs,
    )
    
    pipeline.run_full_pipeline(
        n_demo_episodes=args.demo_episodes,
        airl_timesteps=args.airl_steps,
        airl_config=airl_config,
    )


if __name__ == "__main__":
    main()
