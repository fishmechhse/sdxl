import torch
from typing import List, Tuple
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import pickle
import json

# Импорт вашего SAE
from topksae.topksae import TopKSAESimple
from topksae.topksae import WindowAwareSteeringUpBlockPercent

def to_tokens(x: torch.Tensor) -> torch.Tensor:
    """
    (B, C, H, W) -> (N, C)
    """
    return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])

def get_latents(sae, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        latents = sae.encode(tokens.to(sae.encoder.weight.device))
    return latents.cpu()



def visualize_top_features(
    sae: TopKSAESimple,
    pipe: StableDiffusionXLPipeline,
    feature_indices: List[int],
    strengths: List[float],
    base_prompt: str,
    target_block_path: str,
    output_dir: str,
    trained_window: str = "late_mid"
):
    """
    Визуализирует эффект от усиления/ослабления топ признаков
    """ 
    os.makedirs(output_dir, exist_ok=True)
    
    steering = WindowAwareSteeringUpBlockPercent(sae, 
                                                 target_block_path, 
                                                 trained_window,
                                                 percent_range=(0.3, 0.7),
        num_steps=30)
    steering.register(pipe.unet)
    
    fig, axes = plt.subplots(len(feature_indices), len(strengths), 
                             figsize=(2.5 * len(strengths), 2.5 * len(feature_indices)))
    
    if len(feature_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for i, feat_idx in enumerate(feature_indices):
        for j, strength in enumerate(strengths):
            steering.set_feature(feat_idx, strength)
            generator = torch.Generator(device=pipe.device).manual_seed(42)
            
            image = pipe(
                prompt=base_prompt,
                generator=generator,
                num_inference_steps=30,
                guidance_scale=7.5,
                callback_on_step_end=steering.make_callback(),
                output_type="pil"
            ).images[0]
            
            axes[i, j].imshow(image)
            if i == 0:
                axes[i, j].set_title(f"s={strength}", fontsize=8)
            if j == 0:
                axes[i, j].set_ylabel(f"F{feat_idx}", fontsize=10, rotation=0, ha='right')
            axes[i, j].axis('off')
    
    plt.suptitle(f"Top features for 'smile' concept\nBase prompt: {base_prompt[:50]}...", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_smile_features.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    steering.remove()


def save_results(
    top_features: np.ndarray,
    scores: np.ndarray,
    freq_neutral: np.ndarray,
    output_dir: str
):
    """Сохраняет результаты поиска"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(top_features)
    print(scores)
    print(freq_neutral)
    results = {
        "top_features": top_features,
        "scores": scores.tolist(),
        "freq_on_neutral": freq_neutral,
        "method": "dataset_mining_smile_vs_neutral"
    }
    
    with open(os.path.join(output_dir, "smile_features_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Сохраняем также в pickle
    with open(os.path.join(output_dir, "smile_features_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Результаты сохранены в {output_dir}")

from sklearn.linear_model import LogisticRegression
import numpy as np

def find_smile_features_correct(pipe, sae, layer, smile_prompts, neutral_prompts):

    def collect(prompts):
        feats = []

        def hook(module, input, output):
            if output[0].dim() != 4:
                return

            B = output[0].shape[0] // 2
            delta = output[0][B:] - output[0][:B]

            tokens = delta.permute(0,2,3,1).reshape(-1, delta.shape[1])
            z = sae.encode(tokens.to(sae.encoder.weight.device))

            feats.append(z.detach().cpu())

        h = layer.register_forward_hook(hook)

        for p in prompts:
            pipe(p, num_inference_steps=30, guidance_scale=7.5)

        h.remove()
        return torch.cat(feats, dim=0)

    smile = collect(smile_prompts)
    print(smile.shape)
    neutral = collect(neutral_prompts)
    print(neutral.shape)

    X = torch.cat([smile, neutral], dim=0).numpy()
    y = np.array([1]*len(smile) + [0]*len(neutral))

    clf = LogisticRegression(
        penalty="l2",
        C=0.5,
        max_iter=20
    )

    clf.fit(X, y)

    weights = clf.coef_[0]

    top_features = weights.argsort()[-30:][::-1]

    return top_features, weights

def collect_feature_stats(
    pipe,
    sae,
    layer,
    prompts: List[str],
    num_steps: int = 30,
    percent_range: Tuple[float, float] = (0.3, 0.7),
):
    """
    Собирает средние активации SAE-фичей
    только в заданном диапазоне шагов
    """

    all_latents = []
    current_step = {"value": 0}

    def callback(pipe, step_index, timestep, callback_kwargs):
        current_step["value"] = step_index
        return callback_kwargs

    def should_collect():
        progress = current_step["value"] / num_steps
        return percent_range[0] <= progress <= percent_range[1]
    
    
    # hook
    def hook_fn(module, input, output):
        if not should_collect():
            return
        device = sae.encoder.weight.device

        B = output[0].shape[0] // 2
        uncond = output[0][:B]
        cond = output[0][B:]

        delta = cond - uncond  # CFG direction

        # (B, C, H, W) -> (N_tokens, C)
        tokens = delta.permute(0, 2, 3, 1).reshape(-1, delta.shape[1])

        # SAE encode
        latents = sae.encode(tokens.to(device)).detach().float().cpu()
        

        if isinstance(cond, torch.Tensor) and cond.dim() == 4:
            #tokens = to_tokens(cond)
            latents = get_latents(sae, tokens)
            all_latents.append(latents)

    handle = layer.register_forward_hook(hook_fn)

    # прогон генерации
    for prompt in prompts:
        _ = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=7.5,
            callback_on_step_end=callback,
            output_type="latent"  # быстрее, чем PIL
        )

    handle.remove()

    if len(all_latents) == 0:
        raise RuntimeError("Не удалось собрать активации")

    all_latents = torch.cat(all_latents, dim=0)  # (N, dict_size)

    # фильтр по sparsity
    activation_freq = (all_latents > 0).float().mean(dim=0)
    mask = activation_freq > 0.01

    mean_latents = all_latents.mean(dim=0)

    return mean_latents, mask


def find_smile_features(
    pipe,
    sae,
    target_block_path: str,
    smile_prompts: List[str],
    neutral_prompts: List[str],
    top_k: int = 20,
    num_steps: int = 30,
    percent_range: Tuple[float, float] = (0.3, 0.7),
):
    """
    Находит SAE-фичи, отвечающие за улыбку
    """

    # найти слой
    def find_layer(unet, path):
        layer = unet
        for part in path.split("."):
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    layer = find_layer(pipe.unet, target_block_path)

    print("Сбор статистики (smile)...")
    smile_mean, smile_mask = collect_feature_stats(
        pipe, sae, layer, smile_prompts,
        num_steps=num_steps,
        percent_range=percent_range
    )

    top_features, weights = find_smile_features_correct(pipe, sae, layer, smile_prompts, neutral_prompts)

    print(f"top_features={top_features}, weights={weights}")

    print("Сбор статистики (neutral)...")
    neutral_mean, neutral_mask = collect_feature_stats(
        pipe, sae, layer, neutral_prompts,
        num_steps=num_steps,
        percent_range=percent_range
    )

    # общий mask
    mask = smile_mask & neutral_mask

    # score
    scores = smile_mean - neutral_mean

    # нормализация
    scores = scores / (neutral_mean.std() + 1e-6)

    # применяем mask
    scores_masked = scores.clone()
    scores_masked[~mask] = -1e9

    # top features
    top_pos = torch.topk(scores_masked, top_k).indices
    top_neg = torch.topk(-scores_masked, top_k).indices

    print("Top smile features:")
    print(top_pos.tolist())

    print("Anti-smile features:")
    print(top_neg.tolist())

    return top_pos.tolist(), top_neg.tolist(), scores


def main():
    DATA_PATH = "22_trained_saes_token_norm"  # путь к обученной модели
    DEVICE = "cuda"
    DTYPE = torch.float16

    # Блок и окно для анализа
    TARGET_BLOCK = "up_blocks.0.attentions.0"
    TRAINED_WINDOW = "late_mid"  # для mid_block
    
    # Выходные директории
    OUTPUT_DIR = "smile_feature_analysis_token_norm"
    IMAGES_DIR = os.path.join(OUTPUT_DIR, "generated_images")
    
    print("-" * 70)
    print("ПОИСК ФИЧИ 'УЛЫБКА' МЕТОДОМ АНАЛИЗА ДАТАСЕТА")
    print("-" * 70)
    
    print("Загрузка SAE")
    checkpoint_path = f"{DATA_PATH}/late_mid_best_sae.pt"
    if not os.path.exists(checkpoint_path):
        import glob
        checkpoints = glob.glob(f"{DATA_PATH}/*best_sae.pt")
        if checkpoints:
            checkpoint_path = checkpoints[0]
        else:
            raise FileNotFoundError(f"Не найден checkpoint в {DATA_PATH}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    sae = TopKSAESimple(
        d_model=checkpoint['config']['d_model'],
        dict_size=checkpoint['config']['dict_size'],
        k=checkpoint['config']['k'],
        #auxk=checkpoint['config'].get('auxk', 256),
        #dead_steps_threshold=checkpoint['config'].get('dead_steps_threshold', 2000)
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae = sae.to(DEVICE).to(DTYPE)
    sae.eval()
    
    print(f"   SAE: d_model={sae.d_model}, dict_size={sae.dict_size}, k={sae.k}")
    
    print("Загрузка SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=DTYPE,
        variant="fp16"
    ).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    

    smile_prompts = [
    "a person smiling",
    "portrait of a smiling woman",
     "man with a big smile",
      "man with a smile",
      "girl with a smile",
       "pretty woman with a smile",
       "cat with a smile",
    ]

    neutral_prompts = [
    "a person",
    "portrait of a woman",
    "man face",
    "cat face",
    "dog face",
    ]

    # пример вызова:
    pos_feats, neg_feats, scores = find_smile_features(
     pipe,
     sae,
     target_block_path=TARGET_BLOCK,
     smile_prompts=smile_prompts,
     neutral_prompts=neutral_prompts,
   )
    
    save_results(pos_feats, scores, neg_feats, OUTPUT_DIR)
    
    print("Визуализация топ-5 признаков")
    top_5_features = pos_feats[:3]
    
    base_prompt = "a person"
    strengths = [0, 20, 70]
    
    visualize_top_features(
        sae=sae,
        pipe=pipe,
        feature_indices=top_5_features,
        strengths=strengths,
        base_prompt=base_prompt,
        target_block_path=TARGET_BLOCK,
        output_dir=OUTPUT_DIR,
        trained_window=TRAINED_WINDOW
    )

if __name__ == "__main__":
    main()