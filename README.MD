##  Команда сбора активаций
python3 capture_with_hooks_optimized.py

##  последняя команда запуска на обучение.
python3 train_sae.py --window late_mid --epochs 10 --batch_size 2048 --max_chunks 10

##  Запуск стиринга
python3 steering.py