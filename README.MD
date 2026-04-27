##  последняя команда запуска на обучение. Нужно разнести классы по файлам
python3 22_train_sae.py --window late_mid --epochs 10 --batch_size 2048 --max_chunks 10
-------
##  Top-K Sparse Autoencoder с поддержкой FP16, aux loss и dead feature tracking
python3 train_sae_improved.py --window late_mid --batch_size 1024 --dict_size 16384 --k 64 --auxk 256 --epochs 200 --normalize
python3 train_sae_improved.py --window early_mid --batch_size 2048 --dict_size 16384 --k 64 --auxk 256 --epochs 200