conda activate probegen

python main.py \
  --exp_name=ProbeGen_128__seed_1 \
  --seed=1 \
  --dataset=fmnist_inr \
  \
  --n_tokens=128 \
  --d_hid=256 \
  --mixer_n_layers=6 \
  \
  --gen_type=linear_2_no_acts \
  \
  --batch_size=32 \
  --include_hidden_features=false \
  --lr=0.0003 \
  --epochs=30 \
  --eval_every=500 \
  --n_workers=0 \
  --device=cuda