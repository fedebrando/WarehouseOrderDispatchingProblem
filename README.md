# Warehouse Order Dispatching Problem
Follow next points to find and test your specific best solution (let us indicate the project root with `/`).


1. **Define your warehouse.** First of all, navigate to the `/src` folder and set your warehouse topology in file `initial_state.py`.

1. **Insert your data.** Then, go to `/data` and insert your data or generate them with the script `gen_exp.py`. Data have to be split in three subset (training set, validation set and test set) and each instance have to be represented with the format `t_arrival,pick_zone_idx,drop_zone_idx`. In the beginning of each file CSV, you must indicate the titles for all columns separated by commas.

1. **Set GP hyperparameters.** Set main GP hyperparameters in file `/src/gp_hyperparams.py`. To set more of them, you can modify directly the `/src/gp_main.py` file.

1. **Start tensorboard service.** Before training, run the following command to show real-time information about training.

    ```
    tensorboard --logdir=./runs --samples_per_plugin=text=30 
    ```
    If your text data seem to be subsampled, you can try to increase the last number in the command.

1. **Start training.** Run `/src/gp_main.py` script and wait the end of the GP algorithm.

1. **Testing.** You can test the solution found by copying it from information printed in tensorboard and paste it in the correct place in file `/src/gp_test.py`. The script shows the three objective values of the solution on the test set.

1. **Simulation.** You can also simulate the solution found running `/scr/sim_main.py` after setting the correct policy.

You can find the report and the presentation of the project at `/docs` directory.
