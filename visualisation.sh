python3 make_posterior_surface_plot.py --weight_decay=40 --temperature=1. \
  --dir=runs/surface_plots/mnist_subset160 --dataset_name=mnist \
  --model_name=mlp_classification \
  --checkpoint1=<ckpt1> --checkpoint2=<ckpt2> --checkpoint3=<ckpt3> \  # To replace with actual checkpoints
  --limit_bottom=-0.75 --limit_left=-0.75 --limit_right=1.75 --limit_top=1.75 \
  --grid_size=50