python3 server_policy.py \
--task_name sim_pick_banana \
--ckpt_dir checkpoints/sim_pick_banana/exp17 \
--policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
--seed 0 \
--arm right \
--port 8060 \
