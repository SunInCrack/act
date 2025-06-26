python3 server_policy.py \
--task_name sim_pick_cube \
--ckpt_dir checkpoints/sim_pick_cube/exp0 \
--policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
--seed 0 \
--port 8060 \
