import torch
import torch.nn as nn
from typing import Tuple
import os
import numpy as np
from typing import Tuple, Optional

#############################
####### Текущий результат. Используемые классы
#############################
class TopKSAESimple(nn.Module):
    """Простой Top-K Sparse Autoencoder"""
    
    def __init__(self, d_model: int, dict_size: int, k: int, l1_coef: float = 1e-3):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.k = k
        self.l1_coef = l1_coef
        
        self.encoder = nn.Linear(d_model, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, d_model, bias=True)
        
        self._init_weights()
    
    def loss(self, x: torch.Tensor, recon: torch.Tensor, latents: torch.Tensor) -> dict:
        """Вычисление loss с L1 регуляризацией"""
        mse_loss = nn.functional.mse_loss(recon, x)
        l1_loss = self.l1_coef * torch.mean(torch.abs(latents))
        total_loss = mse_loss + l1_loss
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'l1': l1_loss,
            'sparsity': (latents > 0).float().mean()
        }
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight, a=np.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
            nn.init.zeros_(self.decoder.bias)
        
        self.normalize_decoder()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Кодирование с Top-K"""
        latents = self.encoder(x)
        
        # Top-K активация
        top_k_values, top_k_indices = torch.topk(latents, self.k, dim=-1)
        
        # Создаём разреженный тензор
        sparse_latents = torch.zeros_like(latents)
        sparse_latents.scatter_(-1, top_k_indices, top_k_values)

        sparse_latents = torch.relu(sparse_latents)
        
        return sparse_latents
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        reconstruction = self.decode(latents)
        return reconstruction, latents
    
    # def forward(self, x):
    #    latents, indices, values = self.encode(x)
    #     reconstruction = self.decode(indices, values, x.shape[0])
    #     return reconstruction, latents
    
    @torch.no_grad()
    def normalize_decoder(self):
        norms = self.decoder.weight.data.norm(dim=0, keepdim=True)
        norms = norms.clamp(min=1e-8)
        self.decoder.weight.data = self.decoder.weight.data / norms

class TimestepAwareSAESteeringPercent:

    def __init__(
        self, 
        sae,
        target_block_path: str, 
        trained_window: str = "late_mid",
        device: str = 'cuda',
        percent_range: tuple = None,   # (start, end) в [0,1]
        num_steps: int = 30            # используется для нормализации
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
        
        # атрибуты задающие применение стиринга через маппинг процентов на шаги денойзинга (генерации изображения)
        self.percent_range = percent_range
        self.num_steps = num_steps
        self.use_percent = percent_range is not None
        
        # старый механизм
        self.allowed_steps = self._get_steps_for_window()
        self.current_step = None
        
        print(f"Окно: {trained_window}")
        if self.use_percent:
            print(f"Процентный диапазон: {percent_range}")
        else:
            print(f"Разрешенные шаги: {self.allowed_steps}")
    
    def _get_steps_for_window(self) -> list:
        window_steps = {
            'early': [2, 3, 4],
            'early_mid': [5, 6, 7],
            'late_mid': [8, 9, 10],
            'late': [11, 12, 25],
            'all': list(range(30))
        }
        return window_steps.get(self.trained_window, [8, 9, 10])
    
    def set_feature(self, feature_idx, strength: float = 1.0):
        self.feature_idx = feature_idx
        self.cond_strength = strength
        
        if feature_idx is not None:
            with torch.no_grad():
                self.feature_vector = self.sae.decoder.weight[:, feature_idx].clone()
                self.feature_vector = self.feature_vector / (self.feature_vector.norm() + 1e-8)
                print(f"Фича {feature_idx}, сила={strength}")
        else:
            self.feature_vector = None
    
    def set_current_step(self, step: int):
        self.current_step = step
    
    def should_apply(self) -> bool:
        if self.feature_vector is None or self.cond_strength == 0:
            return False
        
        if self.current_step is None:
            return True
        
        # (проценты)
        if self.use_percent:
            progress = self.current_step / self.num_steps
            start, end = self.percent_range
            return start <= progress <= end
        
        # по конкретным шагам
        return self.current_step in self.allowed_steps
    
    def _find_layer(self, unet):
        parts = self.target_block_path.split('.')
        layer = unet
        
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part, None)
                if layer is None:
                    raise AttributeError(f"Не найден {part}")
        return layer
    
    def _steering_hook_fn(self, module, input, output):
        if not self.should_apply():
            return output
        
        def modify_tensor(tensor):
            if not isinstance(tensor, torch.Tensor):
                return tensor
            
            if tensor.dim() == 4 and tensor.shape[1] == self.sae.d_model:
                B = tensor.shape[0] // 2  # split - это измерение всегда равно 2, так что B = 1
                
                cond_signal = self.cond_strength * self.feature_vector.view(1, -1, 1, 1)
                cond_signal = cond_signal.to(tensor.device, dtype=tensor.dtype)
                
                # split
                uncond = tensor[:B]
                cond   = tensor[B:]
                
                # усиливаем в кондишене и вычитаем в анкондишене
                cond = cond + cond_signal
                uncond = uncond - cond_signal
                
                return torch.cat([uncond, cond], dim=0)
            
            return tensor
        
        if isinstance(output, tuple):
            return tuple(modify_tensor(t) for t in output)
        elif isinstance(output, torch.Tensor):
            return modify_tensor(output)
        
        return output
    
    def register(self, unet):
        self.target_layer = self._find_layer(unet)
        self.hook = self.target_layer.register_forward_hook(self._steering_hook_fn)
        print(f"Hook зарегистрирован на {self.target_block_path}")
        return self
    
    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
    
    def make_callback(self):
        def callback(pipe, step_index, timestep, callback_kwargs):
            self.set_current_step(step_index)
            return callback_kwargs
        
        return callback

class WindowAwareSteeringUpBlockPercent(TimestepAwareSAESteeringPercent):
    """
    Расширенный класс стиринга для обратной совместимости
    """
    pass
