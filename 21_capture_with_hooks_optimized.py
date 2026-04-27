# capture_with_hooks_optimized.py
import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm
import os
import json
from typing import List, Dict, Optional, Callable, Union
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import pandas as pd

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
GEN_SEED = 42
NUM_INFERENCE_STEPS = 30
BASE_OUTPUT_DIR = "sdxl_activations_chunks_optimized"

# Настройки чанков
CHUNK_SIZE_RECORDS = 1000

# Временные окна
TIMESTEPS_WINDOWS = {
    'early': [901, 861, 821],
    'early_mid': [781, 741, 701],
    'late_mid': [661, 621, 581],
    'late': [541, 501, 1]
}
torch.backends.cudnn.benchmark = True


def load_captions_train2017(file_name: str = 'annotations/captions_train2017.json', 
                            num_samples: int = 5000, 
                            random_state: int = 42) -> pd.DataFrame:
    """Загружает captions из COCO JSON файла"""
    import json
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
        df = pd.json_normalize(
            data['annotations'],
            record_path=None,
            meta=['image_id', 'id', 'caption']
        )
        if df.shape[0] < num_samples:
            return df
        return df.sample(n=num_samples, random_state=random_state)


def load_prompts_from_coco(file_name: str = 'annotations/captions_train2017.json', 
                            num_prompts: int = 5000, 
                            random_state: int = 42) -> List[Dict]:
    """Загружает промпты из COCO captions JSON файла"""
    print(f"Загрузка промптов из {file_name}")
    
    df = load_captions_train2017(file_name, num_samples=num_prompts, random_state=random_state)
    
    prompts_data = []
    for idx, row in df.iterrows():
        prompt_text = row.get('caption')
        prompt_image_id = row.get('image_id')
        prompt_id = row.get('id')
        
        if prompt_text and len(str(prompt_text).strip()) > 0:
            prompts_data.append({
                'id': str(prompt_id),
                'image_id': str(prompt_image_id),
                'prompt': str(prompt_text).strip()
            })
    
    print(f"Загружено {len(prompts_data)} промптов")
    
    if prompts_data:
        print("\nПримеры промптов:")
        for i in range(min(3, len(prompts_data))):
            print(f"  {prompts_data[i]['prompt'][:100]}...")
    
    return prompts_data


class OptimizedChunkedSaver:
    """
    Максимально оптимизированный сохранятель - только активации и индексы
    """
    
    def __init__(self, output_dir: str, window_name: str, chunk_size_records: int = 1000, clear_existing: bool = False):
        self.output_dir = output_dir
        self.window_name = window_name
        self.chunk_size_records = chunk_size_records
        
        self.window_dir = os.path.join(output_dir, window_name)
        if clear_existing and os.path.exists(self.window_dir):
            import shutil
            shutil.rmtree(self.window_dir)
        os.makedirs(self.window_dir, exist_ok=True)
        
        self.current_chunk = []  # хранит только активации (numpy arrays)
        self.current_prompt_ids = []  # хранит prompt_id для каждой активации
        self.current_chunk_idx = 0
        self.total_records = 0
        self.existing_chunks = []
        
        self.meta_file = os.path.join(self.window_dir, "metadata.json")
        self._load_metadata()
    
    def _load_metadata(self):
        if os.path.exists(self.meta_file):
            with open(self.meta_file, 'r') as f:
                meta = json.load(f)
                self.total_records = meta.get('total_records', 0)
                self.current_chunk_idx = meta.get('last_chunk_idx', -1) + 1
                self.existing_chunks = meta.get('chunks', [])
    
    def _save_metadata(self):
        metadata = {
            'window_name': self.window_name,
            'chunk_size_records': self.chunk_size_records,
            'total_records': self.total_records,
            'num_chunks': len(self.existing_chunks),
            'last_chunk_idx': self.current_chunk_idx - 1 if self.current_chunk_idx > 0 else -1,
            'activation_shape': list(self.current_chunk[0].shape) if self.current_chunk else None,
            'dtype': 'float16',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'chunks': self.existing_chunks
        }
        with open(self.meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def add(self, activation: np.ndarray, prompt_id: str):
        """Добавляет активацию в текущий чанк"""
        # Конвертируем в float16 сразу
        if activation.dtype != np.float16:
            activation = activation.astype(np.float16)
        
        self.current_chunk.append(activation)
        self.current_prompt_ids.append(prompt_id)
        self.total_records += 1
        
        if len(self.current_chunk) >= self.chunk_size_records:
            self._flush_chunk()
    
    def _flush_chunk(self):
        if not self.current_chunk:
            return
        
        chunk_file = os.path.join(self.window_dir, f"chunk_{self.current_chunk_idx:05d}.npz")
        
        # Сохраняем активации как единый 3D массив (records, height, width)
        # Если активации 2D, добавляем dimension
        activations_array = np.stack(self.current_chunk, axis=0)
        
        # Сохраняем prompt_ids как массив строк
        prompt_ids_array = np.array(self.current_prompt_ids, dtype='U50')
        
        # Сохраняем в NPZ
        np.savez_compressed(
            chunk_file,
            activations=activations_array,
            prompt_ids=prompt_ids_array
        )
        
        file_size_mb = os.path.getsize(chunk_file) / (1024 * 1024)
        
        chunk_info = {
            'chunk_id': self.current_chunk_idx,
            'file': f"chunk_{self.current_chunk_idx:05d}.npz",
            'num_records': len(self.current_chunk),
            'size_mb': round(file_size_mb, 2)
        }
        self.existing_chunks.append(chunk_info)
        self._save_metadata()
        
        print(f"Чанк {self.current_chunk_idx:05d}: {len(self.current_chunk)} записей, {file_size_mb:.1f} MB")
        
        self.current_chunk = []
        self.current_prompt_ids = []
        self.current_chunk_idx += 1
    
    def flush(self):
        if self.current_chunk:
            self._flush_chunk()
    
    def get_total_records(self) -> int:
        return self.total_records
    
    def get_num_chunks(self) -> int:
        return len(self.existing_chunks)


class OptimizedActivationsDataset(Dataset):
    """
    Dataset для загрузки активаций (только активации, без метаданных)
    """
    
    def __init__(self, window_name: str, base_dir: str = BASE_OUTPUT_DIR, 
                 normalize: bool = False, max_chunks: int = None):
        self.window_name = window_name
        self.base_dir = base_dir
        self.normalize = normalize
        self.window_dir = os.path.join(base_dir, window_name)
        
        self.chunk_files = self._get_chunk_files()
        if max_chunks:
            self.chunk_files = self.chunk_files[:max_chunks]
        
        self._build_index()
        
        print(f"Загружен датасет для окна '{window_name}': {len(self)} записей в {len(self.chunk_files)} чанках")
    
    def _get_chunk_files(self) -> List[str]:
        if not os.path.exists(self.window_dir):
            raise FileNotFoundError(f"Директория {self.window_dir} не найдена.")
        
        files = [f for f in os.listdir(self.window_dir) 
                if f.startswith("chunk_") and f.endswith(".npz")]
        files.sort()
        return [os.path.join(self.window_dir, f) for f in files]
    
    def _build_index(self):
        self.cumulative_sizes = []
        self.chunk_sizes = []
        total = 0
        
        for chunk_file in self.chunk_files:
            with np.load(chunk_file, allow_pickle=True) as data:
                size = data['activations'].shape[0]
            
            total += size
            self.cumulative_sizes.append(total)
            self.chunk_sizes.append(size)
        
        self.total_size = total
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        chunk_idx = 0
        while chunk_idx < len(self.cumulative_sizes) and idx >= self.cumulative_sizes[chunk_idx]:
            chunk_idx += 1
        
        prev_size = self.cumulative_sizes[chunk_idx - 1] if chunk_idx > 0 else 0
        inner_idx = idx - prev_size
        
        chunk_file = self.chunk_files[chunk_idx]
        
        with np.load(chunk_file, allow_pickle=True) as data:
            acts = data['activations'][inner_idx]
        
        # Усредняем по spatial dimensions если нужно
        if len(acts.shape) == 3:
            acts = acts.mean(axis=0)  # (h, w, d) -> (d,)
        elif len(acts.shape) == 2:
            acts = acts.flatten()
        
        acts = acts.astype(np.float32)
        
        if self.normalize:
            acts = (acts - acts.mean()) / (acts.std() + 1e-8)
        
        return torch.from_numpy(acts)


class SAEHookCapture:
    """Управление хуками для сбора активаций"""
    
    def __init__(self, target_layers: Dict[str, str]):
        self.target_layers = target_layers
        self.handles = []
        self.current_activation = None
        self.current_step = None
        self.current_timestep = None
        self.target_steps = None
        
    def _make_hook(self):
        def hook(module, input, output):
            if self.target_steps is not None and self.current_step in self.target_steps:
                if isinstance(output, tuple):
                    output = output[0]
                self.current_activation = output.detach().cpu()
        return hook
    
    def register(self, unet_model):
        for layer_path in self.target_layers.values():
            module = self._get_module_by_path(unet_model, layer_path)
            handle = module.register_forward_hook(self._make_hook())
            self.handles.append(handle)
    
    def _get_module_by_path(self, module, path: str):
        parts = path.split('.')
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def set_step_context(self, step: int, timestep: int, target_steps: set):
        self.current_step = step
        self.current_timestep = timestep
        self.target_steps = target_steps
    
    def get_and_clear(self):
        act = self.current_activation
        self.current_activation = None
        return act
    
    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def collect_activations_for_windows(
    windows: List[str],
    num_prompts: int = 5000,
    chunk_size_records: int = 1000,
    output_dir: str = BASE_OUTPUT_DIR,
    random_state: int = 42
) -> Dict[str, OptimizedChunkedSaver]:
    """
    Собирает активации для нескольких временных окон
    """
    if not windows:
        windows = list(TIMESTEPS_WINDOWS.keys())
    
    print(f"{'='*70}")
    print(f"СБОР АКТИВАЦИЙ ДЛЯ {len(windows)} ОКОН")
    print(f"Окна: {', '.join(windows)}")
    print(f"{'='*70}")
    
    prompts_data = load_prompts_from_coco(
        file_name='annotations/captions_train2017.json',
        num_prompts=num_prompts,
        random_state=random_state
    )
    
    if not prompts_data:
        raise ValueError("Не удалось загрузить промпты")
    
    print(f"Загрузка SDXL на {DEVICE}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        variant="fp16" if DTYPE == torch.float16 else None
    ).to(DEVICE)
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    
    target_layers = {
        'up_blocks_0_attentions_0': "up_blocks.0.attentions.0",
    }
    
    savers = {}
    for window_name in windows:
        savers[window_name] = OptimizedChunkedSaver(
            output_dir, window_name, chunk_size_records, clear_existing=True
        )
    
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
    scheduler_timesteps = pipe.scheduler.timesteps.tolist()
    
    windows_target_steps = {}
    all_target_steps = set()
    for window_name in windows:
        timesteps = TIMESTEPS_WINDOWS[window_name]
        target_steps = set()
        for ts in timesteps:
            closest_idx = min(range(len(scheduler_timesteps)), 
                            key=lambda i: abs(scheduler_timesteps[i] - ts))
            target_steps.add(closest_idx)
        windows_target_steps[window_name] = target_steps
        all_target_steps.update(target_steps)
        print(f"Окно '{window_name}': {timesteps} -> steps {sorted(target_steps)}")
    
    hook_capture = SAEHookCapture(target_layers)
    hook_capture.register(pipe.unet)
    generator = torch.Generator(device=DEVICE).manual_seed(GEN_SEED)
    
    for idx, prompt_info in enumerate(tqdm(prompts_data, desc="Processing")):
        prompt_id = prompt_info['id']
        prompt = prompt_info['prompt']
        
        def callback(pipe, step_index, timestep, callback_kwargs):
            hook_capture.set_step_context(step_index, timestep, all_target_steps)
            
            if step_index in all_target_steps:
                activation = hook_capture.get_and_clear()
                if activation is not None:
                    # Убираем batch dimension если есть
                    if len(activation.shape) == 4 and activation.shape[0] == 2:
                        activation = activation[1:2]  # берем второй элемент
                    
                    activation_np = activation.float().cpu().numpy()
                    
                    # Сохраняем для каждого окна, которому принадлежит этот step
                    for window_name in windows:
                        if step_index in windows_target_steps[window_name]:
                            savers[window_name].add(activation_np, prompt_id)
            
            return callback_kwargs
        
        try:
            _ = pipe(
                prompt=prompt,
                generator=generator,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=7.5,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=[],
                output_type="latent"
            )
        except Exception as e:
            print(f"Ошибка: {e}")
            continue
    
    for window_name, saver in savers.items():
        saver.flush()
        print(f"Окно '{window_name}': {saver.get_total_records()} записей, {saver.get_num_chunks()} чанков")
    
    hook_capture.remove()
    return savers

def run_optimized_collection(
    windows: Union[str, List[str]] = 'all',
    num_prompts: int = 6000,
    chunk_size: int = 1000,
    random_state: int = 42
):
    """Запускает сбор активаций"""
    if windows == 'all':
        windows_list = list(TIMESTEPS_WINDOWS.keys())
    elif isinstance(windows, str):
        windows_list = [windows]
    else:
        windows_list = windows
    
    return collect_activations_for_windows(
        windows=windows_list,
        num_prompts=num_prompts,
        chunk_size_records=chunk_size,
        random_state=random_state
    )


if __name__ == '__main__':
    import time
    start_time = time.time()
    
    savers = run_optimized_collection('all', num_prompts=6000, chunk_size=1000)
    
    print(f"Время: {time.time() - start_time:.2f} сек")
    for name, saver in savers.items():
        print(f"  {name}: {saver.get_total_records()} записей")