default:
    just --list

preprocess_data:
    uv run mito_data.py --config config.yaml

train_betaseg:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting betaSeg --base_data_path /home/adhinart/projects/mito_banis/banis/mito_data --save_path ./mito_outputs --devices=1 --sdt
    # uv run BANIS.py --seed 0 --batch_size 4 --n_steps 50000 --data_setting betaSeg --base_data_path /home/adhinart/projects/mito_banis/banis/mito_data --save_path ./mito_outputs --devices=1 --sdt --n_debug_steps 5

train_han24:
    echo "resuming from last checkpoint"
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting han24 --base_data_path /home/adhinart/projects/mito_banis/banis/mito_data --save_path ./mito_outputs --devices=1 --sdt --resume_from_last_checkpoint --exp_name 25-08-14_22-18-41-251762ds_han24_lrng10_s0_b8_mS_k3_lr0.001_wd0.01_schTrue_syn_1.0_drop0.05_shift0.05_intTrue_noise0.5_affine0.5_ns50000_ss128_sdt1_sdtw1.0
    # uv run BANIS.py --seed 0 --batch_size 4 --n_steps 50000 --data_setting han24 --base_data_path /home/adhinart/projects/mito_banis/banis/mito_data --save_path ./mito_outputs --devices=1 --sdt --n_debug_steps 5

launch_all:
    # uv run andromeda_launcher.py train_betaseg
    uv run andromeda_launcher.py train_han24
