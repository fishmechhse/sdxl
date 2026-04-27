import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from datetime import datetime
from topksae.topksae import TopKSAESimple
import torch.nn.functional as F

class SimpleActivationsDataset(torch.utils.data.Dataset):
    """
    Dataset для загрузки пространственных активаций.
    
    Формат исходных данных: (n_records, 1, 1280, 32, 32)
    После обработки: (total_tokens, 1280) - каждый токен как отдельный образец
    """
    
    def __init__(self, window_name: str, base_dir: str = "sdxl_activations_chunks_optimized", 
                 normalize: bool = False, max_chunks: Optional[int] = None):
        """
        Args:
            window_name: имя окна ('early', 'early_mid', 'late_mid', 'late')
            base_dir: базовая директория с данными
            normalize: нормализовать ли активации
            max_chunks: максимальное количество чанков (для тестов)
        """
        self.window_name = window_name
        self.base_dir = base_dir
        self.normalize = normalize
        self.window_dir = os.path.join(base_dir, window_name)
        
        # Получаем список чанков
        print(f"Директория: {self.window_dir}")
        self.chunk_files = self._get_chunk_files()
        self.chunk_files.sort()
        
        if max_chunks:
            self.chunk_files = self.chunk_files[:max_chunks]
        
        # Загружаем все данные
        self._load_all_data()
    
    def _load_all_data(self):
        """
        Загружает и обрабатывает активации.
        
        Трансформация данных:
        - Вход: (n_records, 1, 1280, 32, 32)
        - Шаг 1: squeeze -> (n_records, 1280, 32, 32)
        - Шаг 2: permute + reshape -> (n_records, 1024, 1280)
        - Шаг 3: разворачиваем токены -> (n_records * 1024, 1280)
        
        Результат: self.all_activations.shape = (total_tokens, 1280)
        """
        all_vectors = []  # будет список векторов размерности (1280,)
        channel_norm = False
        token_norm = True
        for chunk_file in tqdm(self.chunk_files, desc=f"Загрузка {self.window_name}"):
            with np.load(chunk_file, mmap_mode='r', allow_pickle=True) as data:
                acts = data['activations']
                # acts.shape: (n_records, 1, 1280, 32, 32)
                
                for i in range(acts.shape[0]):
                    activation = acts[i]
                    # activation.shape: (1, 1280, 32, 32)
                    
                    # Шаг 1: Убираем лишнее измерение
                    # (1, 1280, 32, 32) -> (1280, 32, 32)
                    if len(activation.shape) == 4 and activation.shape[0] == 1:
                        activation = activation.squeeze(0)

                    # activation.shape: (1280, 32, 32)
                    # усредняем по каналам

                    activation = torch.from_numpy(activation)
                    if channel_norm:
                        activation = F.layer_norm(activation, normalized_shape=(32, 32))
                        
                    
                    # Шаг 2: Преобразуем в последовательность токенов
                    # (1280, 32, 32) -> (32, 32, 1280) -> (1024, 1280)
                    activation = activation.permute(1, 2, 0)  # (height, width, channels) -> (32, 32, 1280)
                    activation = activation.reshape(-1, 1280)  # (height*width, channels) -> (1024, 1280)

                    # усредняем по токенам, если поканальное выключено
                    # activation.shape: (1024, 1280)
                    if token_norm:
                        activation = F.layer_norm(activation, normalized_shape=(activation.shape[1],))  
                    all_vectors.append(activation.detach().cpu())     
                    
        self.all_activations = torch.cat(all_vectors, dim=0)
        
        self.d_model = self.all_activations.shape[1]  # 1280
        print(f"Загружено {len(self)} векторов (токенов), размерность {self.d_model}")
    
    def __len__(self):
        """Возвращает количество векторов (токенов)"""
        return len(self.all_activations)
    
    def _get_chunk_files(self) -> List[str]:
        """Возвращает отсортированный список файлов чанков"""
        files = [f for f in os.listdir(self.window_dir) 
                if f.startswith("chunk_") and f.endswith(".npz")]
        files.sort()
        return [os.path.join(self.window_dir, f) for f in files]
    
    def __getitem__(self, idx):
        """
        Возвращает один вектор размерности (d_model,)
        
        Returns:
            torch.Tensor: вектор размерности (1280,)
        """
        return self.all_activations[idx]
        #return torch.from_numpy(self.all_activations[idx])


def train_sae(sae, train_loader, val_loader, epochs, lr, device, save_dir, window_name):
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        sae.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch.float().to(device)
            
            optimizer.zero_grad()
            recon, latents = sae(x)
            losses = sae.loss(x, recon, latents)
            #loss = nn.functional.mse_loss(recon, x)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            sae.normalize_decoder()
            
            #total_loss += loss.item()
            total_loss += losses['total'].item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Val
        sae.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.float().to(device)
                recon, latents = sae(x)
                losses = sae.loss(x, recon, latents)
                #loss = nn.functional.mse_loss(recon, x)
                total_val_loss += losses['total'].item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': sae.state_dict(),
                'config': {'d_model': sae.d_model, 'dict_size': sae.dict_size, 'k': sae.k}
            }, os.path.join(save_dir, f"{window_name}_best_sae.pt"))
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    return train_losses, val_losses


def analyze_features(sae, dataset, window_name, save_dir):
    sae.eval()
    device = next(sae.parameters()).device
    print("111")
    data = torch.from_numpy(dataset.all_activations).float().to(device)
    print("222")
    all_latents = []
    all_sparsity = []
    with torch.no_grad():
        for i in range(0, len(data), 4096):
            batch = data[i:i+4096]
            latents = sae.encode(batch)
            print(f"latents.shape={latents.shape}")
            all_latents.append(latents.cpu().numpy().astype(np.float16))
    print(f"{len(all_latents)}")
    latents_all = np.concatenate(all_latents, axis=0)
    print("444")
    freq = (latents_all > 0).mean(axis=0)
    mean_act = latents_all.mean(axis=0)
    print("555")
    top_freq = np.argsort(freq)[::-1][:20]
    top_mean = np.argsort(mean_act)[::-1][:20]
    print("666")
    print(f"Топ-20 по частоте")
    for i, idx in enumerate(top_freq):
        print(f"  {i+1}. Feature {idx}: freq={freq[idx]:.4f}, mean={mean_act[idx]:.4f}")
    
    print(f"Топ-20 по средней активации")
    for i, idx in enumerate(top_mean):
        print(f"  {i+1}. Feature {idx}: mean={mean_act[idx]:.4f}, freq={freq[idx]:.4f}")
    print("666")
    np.savez(os.path.join(save_dir, f"{window_name}_top_features.npz"),
             top_by_freq=top_freq, top_by_mean=top_mean,
             feature_frequency=freq, feature_mean_activation=mean_act)
    
    return top_freq, top_mean

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=str, default='late')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dict_size', type=int, default=16384)
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_chunks', type=int, default=2)
    
    args = parser.parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = "22_trained_saes_token_norm"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print(f"Save dir: {SAVE_DIR}")
    
    dataset = SimpleActivationsDataset(
        window_name=args.window,
        base_dir="sdxl_activations_chunks_optimized",
        normalize=True,
        max_chunks=args.max_chunks
    )
    
    total = len(dataset)
    train_size = int(0.9 * total)
    val_size = total - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    sae = TopKSAESimple(d_model=1280, dict_size=args.dict_size, k=args.k).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in sae.parameters()):,}")
    print(f"lr: {args.lr}")
    
    train_losses, val_losses = train_sae(
        sae, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr,
        device=DEVICE, save_dir=SAVE_DIR, window_name=args.window
    )
    
    print("Анализ")
    #analyze_features(sae, dataset, args.window, SAVE_DIR)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.savefig(os.path.join(SAVE_DIR, f"{args.window}_loss.png"))
    plt.show()