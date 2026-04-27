# steering_with_timesteps.py
import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import os
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import numpy as np
from tqdm import tqdm

# ============================================
# КЛАСС SAE
# ============================================

from topksae.topksae import TopKSAESimple
from topksae.topksae import WindowAwareSteeringUpBlockPercent


class TimestepAwareSAESteering:
    """
    SAE Steering с поддержкой конкретных шагов денойзинга для каждого окна
    """
    
    def __init__(
        self, 
        sae: TopKSAESimple, 
        target_block_path: str, 
        trained_window: str,  # "early", "early_mid", "late_mid", "late"
        device: str = 'cuda'
    ):
        self.sae = sae
        self.target_block_path = target_block_path
        self.trained_window = trained_window
        self.device = device
        
        self.feature_idx = None
        self.cond_strength = 0.0
        self.uncond_strength = 0.0
        self.feature_vector = None
        self.hook = None
        self.target_layer = None
        
        # Получаем шаги для данного окна
        self.allowed_steps = self._get_steps_for_window()
        self.current_step = None
        
        print(f"📊 Окно: {trained_window}")
        print(f"   Разрешенные шаги: {self.allowed_steps}")
    
    def _get_steps_for_window(self) -> list:
        """Возвращает шаги денойзинга для каждого окна"""
        window_steps = {
            'early': [2, 3, 4],
            'early_mid': [5, 6, 7],
            'late_mid': [8, 9, 10],
            'late': [11, 12, 25],
            'all': list(range(30))  # все шаги
        }
        return window_steps.get(self.trained_window, [8, 9, 10])
    
    def set_feature(self, feature_idx: Optional[int], strength: float = 1.0):
        """Установить фичу для стиринга"""
        self.feature_idx = feature_idx
        self.cond_strength = strength
        
        if feature_idx is not None:
            with torch.no_grad():
                self.feature_vector = self.sae.decoder.weight[:, feature_idx].clone()
                # Нормализуем вектор для стабильности
                self.feature_vector = self.feature_vector / (self.feature_vector.norm() + 1e-8)
                print(f"🎯 Фича {feature_idx}, сила={strength}, норма={self.feature_vector.norm().item():.4f}")
        else:
            self.feature_vector = None
    
    def set_current_step(self, step: int):
        """Устанавливает текущий шаг денойзинга"""
        self.current_step = step
    
    def should_apply(self) -> bool:
        """Проверяет, нужно ли применять фичу на текущем шаге"""
        if self.feature_vector is None or self.cond_strength == 0:
            return False
        
        if self.current_step is None:
            return True
        
        return self.current_step in self.allowed_steps
    
    def _find_layer(self, unet):
        """Поиск целевого слоя в UNet"""
        parts = self.target_block_path.split('.')
        layer = unet
        
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part, None)
                if layer is None:
                    raise AttributeError(f"Не найден атрибут {part} в {type(layer)}")
        return layer
    
    def _steering_hook_fn(self, module, input, output):
        """
        Hook функция для модификации активаций
        Применяется только на разрешенных шагах
        """
        #print(f"len(output)={len(output)}")
        #print(f"output[0].shape={output[0].shape}")

        if not self.should_apply():
            #print("not should_apply")
            return output
        
        def modify_tensor(tensor):
            if not isinstance(tensor, torch.Tensor):
                return tensor
            
            if tensor.dim() == 4 and tensor.shape[1] == self.sae.d_model:
                cond_signal = self.cond_strength * self.feature_vector.view(1, -1, 1, 1)
                uncond_signal = self.uncond_strength * self.feature_vector.view(1, -1, 1, 1)
                
                cond_signal = cond_signal.to(tensor.device, dtype=tensor.dtype)
                uncond_signal = uncond_signal.to(tensor.device, dtype=tensor.dtype)
                #print(f"uncond_signal.shape={uncond_signal}")
                #print(f"cond_signal.shape={cond_signal.shape}")
                tensor[0] = tensor[0] - cond_signal[0] # активации полученные по нулевому CFG
                tensor[1] = tensor[1] + cond_signal[0]

                #steering_signal = steering_signal.to(tensor.device, dtype=tensor.dtype)
                #return tensor + steering_signal
                #tensor = torch.cat([uncond, cond], dim=0)
                return tensor
            return tensor
        
        if isinstance(output, tuple):
            modified_output = tuple(modify_tensor(t) for t in output)
            return modified_output
        elif isinstance(output, torch.Tensor):
            return modify_tensor(output)
        return output
    
    def register(self, unet):
        """Регистрация hook на UNet"""
        self.target_layer = self._find_layer(unet)
        self.hook = self.target_layer.register_forward_hook(self._steering_hook_fn)
        print(f"Hook зарегистрирован на {self.target_block_path}")
        return self
    
    def remove(self):
        """Удаление hook"""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
            print(f"Hook удален с {self.target_block_path}")
    
    def make_callback(self):
        """
        callback функция для отслеживания шагов денойзинга
        """
        def callback(pipe, step_index, timestep, callback_kwargs):
            # сеттим текущий шаг (step_index начинается с 0)
            # шаги от 0 до 29 включительно
            self.set_current_step(step_index)
            return callback_kwargs
        
        return callback


def print_separator():
    print(f"{'-'*70}")

def steering_grid_generation(
    sae: TopKSAESimple,
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    feature_indices: List[int],
    strengths: List[float],
    target_block_path: str,
    trained_window: str,
    output_dir: str,
    seed: int = 42,
    num_inference_steps: int = 30,
    file_prefix: str = "",
    guidance_scale: float = 7.5
):
    """
    Генерация сетки изображений для разных фич и сил интервенции
    С учетом разрешенных шагов для каждого окна
    """
    os.makedirs(output_dir, exist_ok=True)
    steering = WindowAwareSteeringUpBlockPercent(
        sae=sae,
        target_block_path=target_block_path,
        trained_window=trained_window,
        device=pipe.device,
        percent_range=(0.5, 0.8),
        num_steps=num_inference_steps
    )
    steering.register(pipe.unet)
    
    all_images = []
    
    for feat_idx in tqdm(feature_indices, desc="Обработка фич"):
        print(f"Фича {feat_idx}")
        print(f"Разрешенные шаги: {steering.allowed_steps}")
        feature_images = []
        
        for strength in strengths:
            print(f"Сила: {strength}")
            steering.set_feature(feat_idx, strength)
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
            
            try:
                image = pipe(
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    callback_on_step_end=steering.make_callback(),
                    output_type="pil"
                ).images[0]
                feature_images.append(image)
                print(f"Готово")
            except Exception as e:
                print(f"Ошибка: {e}")
                from PIL import Image
                feature_images.append(Image.new('RGB', (512, 512), color='black'))
        
        all_images.append(feature_images)
    
    # Сохраняем сетку
    try:
        n_features = len(feature_indices)
        n_strengths = len(strengths)
        
        fig, axes = plt.subplots(n_features, n_strengths, 
                                 figsize=(2.5 * n_strengths, 2.5 * n_features))
        
        if n_features == 1:
            axes = axes.reshape(1, -1)
        if n_strengths == 1:
            axes = axes.reshape(-1, 1)
        
        for i, feat_idx in enumerate(feature_indices):
            for j, (img, strength) in enumerate(zip(all_images[i], strengths)):
                axes[i, j].imshow(img)
                if i == 0:
                    axes[i, j].set_title(f"s={strength}", fontsize=8)
                if j == 0:
                    axes[i, j].set_ylabel(f"F{feat_idx}", fontsize=10, rotation=0, ha='right')
                axes[i, j].axis('off')
        
        plt.suptitle(
            f"Steering: {prompt}\n"
            f"Window: {trained_window}, Steps: {steering.allowed_steps}\n"
            f"Block: {target_block_path.split('.')[-1]}", 
            fontsize=12, y=1.02
        )
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"grid_{trained_window}_{file_prefix}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Сетка сохранена: {output_path}")
        
    except Exception as e:
        print(f"Ошибка при сохранении сетки: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all')
    
    print("Генерация оригинального изображения")
    steering.set_feature(None, 0)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    try:
        original_image = pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            callback_on_step_end=steering.make_callback(),
            output_type="pil"
        ).images[0]
        
        original_path = os.path.join(output_dir, f"original_{file_prefix}.png")
        original_image.save(original_path)
        print(f"Оригинал сохранен: {original_path}")
    except Exception as e:
        print(f"Ошибка при генерации оригинала: {e}")
        original_image = None
    
    steering.remove()
    return all_images, original_image


def test_all_windows(
    sae: TopKSAESimple,
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    feature_idx: int,
    strength: float,
    target_block_path: str,
    output_dir: str,
    seed: int = 42,
    num_inference_steps: int = 30
):
    """
    Тестирование одной фичи на разных временных окнах
    """
    windows = ['early', 'early_mid', 'late_mid', 'late']
    results = {}
    
    for window in windows:
        print_separator()
        print(f"Тестирование окна: {window}")
        print_separator()
        
        steering = WindowAwareSteeringUpBlockPercent(
            sae=sae,
            target_block_path=target_block_path,
            trained_window=window,
            device=pipe.device,
            percent_range=(0.5, 0.7),
            num_steps=num_inference_steps
        )
        steering.register(pipe.unet)
        steering.set_feature(feature_idx, strength)
        
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        
        image = pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            callback_on_step_end=steering.make_callback(),
            output_type="pil"
        ).images[0]
        
        results[window] = image
        steering.remove()
    
    # Визуализация результатов для всех окон
    fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4))
    
    for i, (window, img) in enumerate(results.items()):
        axes[i].imshow(img)
        axes[i].set_title(f"{window}\nSteps: {steering._get_steps_for_window()}", fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle(f"Feature {feature_idx}, Strength={strength}\n{prompt}", fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"window_comparison_f{feature_idx}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Сравнение окон сохранено: {output_path}")
    return results


def steering_grid_generation_prompts(
    sae: TopKSAESimple,
    pipe: StableDiffusionXLPipeline,
    prompts: List[str],  # Список промптов (по одному в строке)
    feature_idx: int,     # Одна фича
    strengths: List[float],
    target_block_path: str,
    trained_window: str,
    output_dir: str,
    seed: int = 42,
    num_inference_steps: int = 30,
    file_prefix: str = "",
    guidance_scale: float = 7.5
):
    """
    Генерация сетки изображений для одной фичи и разных промптов
    - Каждый промпт в отдельной строке
    - Разные силы интервенции в колонках
    
    Args:
        prompts: список промптов (каждый промпт - новая строка)
        feature_idx: индекс фичи (одна и та же для всех промптов)
        strengths: список сил интервенции (колонки)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем steering
    steering = WindowAwareSteeringUpBlockPercent(
        sae=sae,
        target_block_path=target_block_path,
        trained_window=trained_window,
        device=pipe.device,
        percent_range=(0.5, 0.8),
        num_steps=num_inference_steps
    )
    steering.register(pipe.unet)
    
    # Результаты: список списков [промпт][сила]
    all_images = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Обработка промптов")):
        print(f"Промпт {i+1}: {prompt[:50]}...")
        prompt_images = []
        
        for strength in strengths:
            print(f"Сила: {strength}")
            steering.set_feature(feature_idx, strength)
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
            
            try:
                image = pipe(
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    callback_on_step_end=steering.make_callback(),
                    output_type="pil"
                ).images[0]
                prompt_images.append(image)
                print(f"Готово")
            except Exception as e:
                print(f"Ошибка: {e}")
                from PIL import Image
                prompt_images.append(Image.new('RGB', (512, 512), color='black'))
        
        all_images.append(prompt_images)
    
    # Сохраняем сетку
    try:
        n_prompts = len(prompts)
        n_strengths = len(strengths)
        
        # Создаем фигуру с соотношением сторон
        fig, axes = plt.subplots(n_prompts, n_strengths, 
                                 figsize=(2.5 * n_strengths, 2.5 * n_prompts))
        
        # Если только один промпт - преобразуем в 2D
        if n_prompts == 1:
            axes = axes.reshape(1, -1)
        if n_strengths == 1:
            axes = axes.reshape(-1, 1)
        
        # Заполняем сетку
        for i, prompt in enumerate(prompts):
            for j, (img, strength) in enumerate(zip(all_images[i], strengths)):
                axes[i, j].imshow(img)
                if i == 0:
                    axes[i, j].set_title(f"s={strength}", fontsize=8)
                if j == 0:
                    # Показываем короткую версию промпта
                    short_prompt = prompt[:30] + "..." if len(prompt) > 30 else prompt
                    axes[i, j].set_ylabel(short_prompt, fontsize=8, rotation=45, ha='right', va='center')
                axes[i, j].axis('off')
        
        plt.suptitle(
            f"Steering: Feature {feature_idx}\n"
            f"Window: {trained_window}, Steps: {steering.allowed_steps}\n"
            f"Block: {target_block_path.split('.')[-1]}", 
            fontsize=12, y=1.02
        )
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"grid_prompts_{trained_window}_f{feature_idx}_{file_prefix}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Сетка сохранена: {output_path}")
        
    except Exception as e:
        print(f"Ошибка при сохранении сетки: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all')
    
    # Генерируем оригинальные изображения (без стиринга) для каждого промпта
    print("Генерация оригинальных изображений")
    steering.set_feature(None, 0)
    original_images = []
    
    for i, prompt in enumerate(prompts):
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        
        try:
            original_image = pipe(
                prompt=prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                callback_on_step_end=steering.make_callback(),
                output_type="pil"
            ).images[0]
            original_images.append(original_image)
            
            # Сохраняем отдельно каждое оригинальное изображение
            original_path = os.path.join(output_dir, f"original_{file_prefix}_prompt{i+1}.png")
            original_image.save(original_path)
            print(f"Оригинал сохранен: {original_path}")
        except Exception as e:
            print(f"Ошибка при генерации оригинала для {prompt}: {e}")
            from PIL import Image
            original_images.append(Image.new('RGB', (512, 512), color='black'))
    
    steering.remove()
    return all_images, original_images


def steering_grid_generation_prompts_comparison(
    sae: TopKSAESimple,
    pipe: StableDiffusionXLPipeline,
    prompts: List[str],
    feature_indices: List[int],  # Несколько фич
    strengths: List[float],
    target_block_path: str,
    trained_window: str,
    output_dir: str,
    seed: int = 42,
    num_inference_steps: int = 30,
    file_prefix: str = "",
    guidance_scale: float = 7.5
):
    """
    Сравнение нескольких фич на нескольких промптах
    - Строки: промпты
    - Столбцы: фичи (с разными силами интервенции)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    steering = WindowAwareSteeringUpBlockPercent(
        sae=sae,
        target_block_path=target_block_path,
        trained_window=trained_window,
        device=pipe.device,
        percent_range=(0.5, 0.8),
        num_steps=num_inference_steps
    )
    steering.register(pipe.unet)
    
    # Результаты: [промпт][фича][сила]
    all_images = []
    best_strength = strengths[len(strengths)//2] if strengths else strengths[0]
    
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Обработка промптов")):
        print(f"Промпт {prompt_idx+1}: {prompt[:50]}...")
        prompt_features = []
        
        for feat_idx in feature_indices:
            print(f"Фича {feat_idx}, сила={best_strength}")
            steering.set_feature(feat_idx, best_strength)
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
            
            try:
                image = pipe(
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    callback_on_step_end=steering.make_callback(),
                    output_type="pil"
                ).images[0]
                prompt_features.append(image)
            except Exception as e:
                print(f"Ошибка: {e}")
                from PIL import Image
                prompt_features.append(Image.new('RGB', (512, 512), color='black'))
        
        all_images.append(prompt_features)
    
    # Сохраняем сетку
    n_prompts = len(prompts)
    n_features = len(feature_indices)
    
    fig, axes = plt.subplots(n_prompts, n_features, 
                             figsize=(2.5 * n_features, 2.5 * n_prompts))
    
    if n_prompts == 1:
        axes = axes.reshape(1, -1)
    if n_features == 1:
        axes = axes.reshape(-1, 1)
    
    for i, prompt in enumerate(prompts):
        for j, (feat_idx, img) in enumerate(zip(feature_indices, all_images[i])):
            axes[i, j].imshow(img)
            if i == 0:
                axes[i, j].set_title(f"F{feat_idx}", fontsize=10)
            if j == 0:
                short_prompt = prompt[:30] + "..." if len(prompt) > 30 else prompt
                axes[i, j].set_ylabel(short_prompt, fontsize=8, rotation=45, ha='right')
            axes[i, j].axis('off')
    
    plt.suptitle(
        f"Steering Comparison\n"
        f"Window: {trained_window}, Strength={best_strength}\n"
        f"Block: {target_block_path.split('.')[-1]}", 
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"grid_comparison_{trained_window}_{file_prefix}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    steering.remove()
    return all_images



if __name__ == "__main__":
    
    # Конфигурация
    #DATA_PATH = "trained_saes_20260425_015051"
    #DATA_PATH = "22_trained_saes"
    DATA_PATH = "22_trained_saes_token_norm"
    CHECKPOINT_PATH = "late_mid_best_sae.pt"
    FEATURES_FILE = "late_mid_top_features.npz"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    DTYPE = torch.float16
    OUT_DIR = "steering_grid_results_norm"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print_separator()
    print("STEERING WITH TIMESTEP-AWARE SAE")
    print_separator()
    print(f"Device: {DEVICE}")
    print("Загрузка SAE")
    
    checkpoint_path = f"{DATA_PATH}/{CHECKPOINT_PATH}"
    if not os.path.exists(checkpoint_path):
        import glob
        checkpoints = glob.glob(f"{DATA_PATH}/*best_sae.pt")
        if checkpoints:
            checkpoint_path = checkpoints[0]
            print(f"Найден checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Не найден checkpoint в {DATA_PATH}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    sae = TopKSAESimple(
        d_model=checkpoint['config']['d_model'],
        dict_size=checkpoint['config']['dict_size'],
        k=checkpoint['config']['k']
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae = sae.to(DEVICE).to(DTYPE)
    sae.eval()
    
    print(f"SAE: d_model={sae.d_model}, dict_size={sae.dict_size}, k={sae.k}")
    
    features_path = f"{DATA_PATH}/{FEATURES_FILE}"
    
    def load_features_npz(filepath):
        with np.load(filepath, allow_pickle=True) as data:
            top_by_freq = data['top_by_freq']
            top_by_mean = data['top_by_mean']
            return top_by_freq, top_by_mean
    
    if os.path.exists(features_path):
        top_by_freq, top_by_mean = load_features_npz(features_path)
        test_features = top_by_freq[:5].tolist()
        print(f"Тестируемые фичи: {test_features}")
    else:
        test_features = [4251,  2475,  4714,  2130,  4639,  1298]
        print(f"Используем стандартные фичи: {test_features}")
    
    print("Загрузка SDXL pipeline")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=DTYPE,
    ).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None  

    
    STRENGTHS = [0, 25, 50, 75]
    #STRENGTHS = [0, 50, 100, 150]
    TARGET_BLOCK = "up_blocks.0.attentions.0"
    TRAINED_WINDOW = "late_mid"  # "early", "early_mid", "late_mid", "late"
    
    one_feature = True
    if not one_feature:
      # 4. Параметры стиринга
      PROMPTS = {
       # "man": "portrait of a man with detailed face",
       # "woman": "a young woman",
       "woman_frek": "young woman with freckles high resolution, fine details, professional shot",
      }
      print(f"Настройки стиринга:")
      print(f"Окно: {TRAINED_WINDOW}")
      print(f"Шаги: {WindowAwareSteeringUpBlockPercent(sae, TARGET_BLOCK, TRAINED_WINDOW, DEVICE)._get_steps_for_window()}")
      print(f"Блок: {TARGET_BLOCK}")
    
      # 5. Генерация
      for file_prefix, prompt in PROMPTS.items():
        print(f"{'-'*70}")
        print(f"Промпт: {prompt}")
        print(f"{'-'*70}")
        
        images, original = steering_grid_generation(
            sae=sae,
            pipe=pipe,
            prompt=prompt,
            feature_indices=test_features,
            strengths=STRENGTHS,
            target_block_path=TARGET_BLOCK,
            trained_window=TRAINED_WINDOW,
            output_dir=OUT_DIR,
            seed=42,
            num_inference_steps=30,
            file_prefix=file_prefix,
            guidance_scale=7.5
        )
    
      print_separator()
      print(f"Результаты: {OUT_DIR}")

    else:
      PROMPTS_LIST = [
        "portrait of a man with detailed face, professional photography",
        "portrait of a young woman, high resolution, fine details",
        "a cat on a sofa, realistic",
        "a pirate in a hat",
        "a mister smith",
        "angry man"
      ]
    
      FEATURE_IDX = 2475  # Одна фича
      STRENGTHS = [-10, 0, 10, 25, 50, 75, 100]
    
      print_separator()
      print("ВАРИАНТ 1: Одна фича на разных промптах")
      print_separator()
      print(f"Фича: {FEATURE_IDX}")
      print(f"Промптов: {len(PROMPTS_LIST)}")
      print(f"Силы: {STRENGTHS}")
    
      images, originals = steering_grid_generation_prompts(
        sae=sae,
        pipe=pipe,
        prompts=PROMPTS_LIST,
        feature_idx=FEATURE_IDX,
        strengths=STRENGTHS,
        target_block_path=TARGET_BLOCK,
        trained_window=TRAINED_WINDOW,
        output_dir=OUT_DIR,
        seed=42,
        num_inference_steps=30,
        file_prefix="single_feature",
        guidance_scale=7.5
      )
