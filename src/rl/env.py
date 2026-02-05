import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from states.game_state import GameState
from settings import *

class GetMyWeedEnv(gym.Env):
    def __init__(self, brain_stack=None):
        super(GetMyWeedEnv, self).__init__()
        
        # 3 actions Run, Jump, Fast Fall
        self.action_space = spaces.Discrete(3)
        
        # 7 inputs pour le reseau (position, vitesse, obstacles...)
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(7,), 
            dtype=np.float32
        )
        self.game = None
        self.brain = brain_stack

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # On repart a zero sur une nouvelle game
        self.game = GameState(self.brain)
        self.game.player.ai_mode = True
        return self._get_obs(), {}

    def step(self, action):
        # Applique l'action et avance la physique
        self.game.player.current_ai_action = action
        self.game.update(1.0/60.0, [])
        
        # Calcul des rewards
        reward = 0.0
        p = self.game.player
        
        reward += 0.1 # Juste pour survivre
        
        # --- BONUS ---
        if p.just_collected_weed:
            reward += 5.0 # L'objectif principal
        if p.just_hit_enemy:
            reward += 2.0 # Boing !
        if p.just_dodged_enemy:
            reward += 1.0 # Matrix style
        if p.just_landed:
            reward += 0.2 
        if p.just_fast_fell:
            reward += 0.1 # Style points
        if p.just_used_shield:
            reward += 0.5
        if p.just_used_magnet:
            reward += 0.5
        if p.just_reached_max_combo:
            reward += 3.0 # C-C-Combo !
        if p.just_reached_min_withdrawal:
            reward += 2.0 # Clean
        if p.just_reached_max_speed:
            reward += 0.5
        if p.just_reached_max_hp:
            reward += 1.0

        # --- MALUS ---
        if p.just_jumped:
            reward -= 0.01 # Petit coût pour éviter le spam
        if p.just_took_damage:
            reward -= 5.0
        if p.just_died:
            reward -= 50.0
        if p.just_reached_max_withdrawal:
            reward -= 5.0 # OD imminent
        if p.just_reached_min_speed:
            reward -= 0.5 # Trop lent
        if p.just_reached_min_hp:
            reward -= 2.0 # Danger
        if p.just_reached_min_combo:
            reward -= 0.5 # Combo cassé
            
        terminated = False
        if p.hp <= 0 or p.rect.top > DEATH_Y:
            terminated = True
            
        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        scan = self.game.scan_surroundings()
        p = self.game.player
        
        # Normalisation des entrees pour que le reseau capte mieux
        obs = np.array([
            p.rect.centery / SCREEN_HEIGHT,
            np.clip(p.velocity_y / 1000.0, -1, 1),
            np.clip(scan['next_gap_dist'] / 1500.0, 0, 1),
            np.clip(scan['next_enemy_dist'] / 1500.0, 0, 1),
            scan['next_enemy_type'] / 2.0,
            1.0 if p.on_ground else 0.0,
            p.speed / 1000.0
        ], dtype=np.float32)
        return obs

    def render(self):
        pass