echo Running HMC on MNIST...
python3 run_hmc.py --seed=42 --weight_decay=1. --temperature=1. \
  --dir=runs/hmc/mnist_subset160 --dataset_name=mnist \
  --model_name=mlp_classification --step_size=3.e-5 --trajectory_len=1.5 \
  --num_iterations=100 --max_num_leapfrog_steps=50000 \
  --num_burn_in_iterations=10 --subset_train_to=160

echo Running SGMCMC on MNIST...
python3 run_sgmcmc.py --seed=42 --weight_decay=1. \
  --dir=runs/sgmcmc/mnist_subset160 --dataset_name=mnist \
  --model_name=mlp_classification --init_step_size=3.e-5 \
  --num_epochs=10000 --num_burnin_epochs=1000 --step_size_schedule=cyclical \
  --step_size_cycle_length_epochs=50 --ensemble_freq=50 --eval_freq=10 \
  --batch_size=160 --save_freq=1000 --subset_train_to=160 \
  --preconditioner=None --momentum=0.95 --eval_freq=10 --save_all_ensembled

echo Running SGD on MNIST...
python3 run_sgd.py --seed=42 --weight_decay=1. \
  --dir=runs/sgd/mnist_subset160 --dataset_name=mnist \
  --model_name=mlp_classification --step_size=3.e-5 \
  --num_epochs=500 --eval_freq=10 --batch_size=80 \
  --save_freq=500 --subset_train_to=160