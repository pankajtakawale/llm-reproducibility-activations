mkdir -p results_backup
mv results/* results_backup/

# Phase 1: Quick validation (5 min)
# config_cpu.py: trials=3, iters=100
#python run_all_experiments.py --models charlm tinylstm --activations relu gelu
#python process_results.py

# Phase 2: CPU comprehensive (20 min)
# config_cpu.py: trials=3, iters=200
python run_all_experiments.py --models all
python process_results.py

# Phase 3: GPU publication (2-3 hours on cluster)
# config.py: trials=5, iters=5000
# Transfer code to GPU cluster
#python run_all_experiments.py --models all
#python process_results.py