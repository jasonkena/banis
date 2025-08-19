default:
    just --list

train_base:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting base --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --devices=1

train_liconn:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting liconn --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --devices=1

train_multichannel:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting multichannel --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --devices=1

train_neg_guidance:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting neg_guidance --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --devices=1

train_pos_guidance:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting pos_guidance --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --devices=1

train_slice_perturbed:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting slice_perturbed --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --devices=1

train_touching_thin:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting touching_thin --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --devices=1

train_100:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting train_100 --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --devices=1

train_no_touch_thick:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting no_touch_thick --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --devices=1

train_base_sdt:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting base --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --sdt --devices=1 #--n_debug_steps 5

train_base_long:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 200000 --data_setting base --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --devices=1

train_base_long_long:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 1000000 --data_setting base --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --devices=1

resume_train_base_long:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 200000 --data_setting base --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --checkpoint /home/adhinart/projects/nisb2/banis/outputs/25-07-18_11-37-53-618611ds_base_lrng10_s0_b8_mS_k3_lr0.001_wd0.01_schTrue_syn_1.0_drop0.05_shift0.05_intTrue_noise0.5_affine0.5_ns200000_ss128/lightning_logs/version_0/checkpoints/last.ckpt --devices=1


generate_all_affs:
    uv run generate_all_affs.py --checkpoint_path /home/adhinart/projects/nisb2/banis/outputs/25-06-24_12-58-58-562530ds_base_lrng10_s0_b8_mS_k3_lr0.001_wd0.01_schTrue_syn_1.0_drop0.05_shift0.05_intTrue_noise0.5_affine0.5_ns50000_ss128/lightning_logs/version_0/checkpoints/last.ckpt

generate_all_affs_long:
    uv run generate_all_affs.py --checkpoint_path /home/adhinart/projects/nisb2/banis/outputs/25-07-18_11-37-53-618611ds_base_lrng10_s0_b8_mS_k3_lr0.001_wd0.01_schTrue_syn_1.0_drop0.05_shift0.05_intTrue_noise0.5_affine0.5_ns200000_ss128/lightning_logs/version_0/checkpoints/last.ckpt

generate_all_affs_sdt:
    uv run generate_all_affs.py --checkpoint_path /home/adhinart/projects/nisb2/banis/outputs/25-08-14_22-05-31-824882ds_base_lrng10_s0_b8_mS_k3_lr0.001_wd0.01_schTrue_syn_1.0_drop0.05_shift0.05_intTrue_noise0.5_affine0.5_ns50000_ss128_sdt1_sdtw1.0/default/checkpoints/last.ckpt --prediction_channels=7

launch_all:
    ## uv run andromeda_launcher.py train_base
    ## uv run andromeda_launcher.py train_no_touch_thick
    # uv run andromeda_launcher.py train_liconn
    # uv run andromeda_launcher.py train_multichannel
    # uv run andromeda_launcher.py train_neg_guidance
    # uv run andromeda_launcher.py train_pos_guidance
    # uv run andromeda_launcher.py train_slice_perturbed
    # uv run andromeda_launcher.py train_touching_thin
    # uv run andromeda_launcher.py train_100
    # uv run andromeda_launcher.py train_base_long
    uv run andromeda_launcher.py train_base_sdt
