import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Petit hack pour que Python trouve les autres modules (settings, states...)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rl.env import GetMyWeedEnv

# Config
MODELS_DIR = "models/PPO"
LOG_DIR = "logs"
TIMESTEPS = 100000 # On peut augmenter si besoin

def train():
    # Création des dossiers si pas là
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Init de l'environnement
    # On wrap dans DummyVecEnv pour la compatibilité SB3 (et perfs si on en mettait plusieurs)
    env = DummyVecEnv([lambda: GetMyWeedEnv()])

    # Création du modèle
    # MlpPolicy car on a des inputs numériques simples (pas d'images/CNN)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

    # Callback pour sauver régulièrement (tous les 10k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=MODELS_DIR,
        name_prefix="weed_bot"
    )

    print("🚀 Lancement de l'entrainement...")
    
    # On lance le training
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    # Save final
    model.save(f"{MODELS_DIR}/weed_bot_final")
    print("✅ Entrainement terminé et modèle sauvegardé !")

if __name__ == "__main__":
    train()
