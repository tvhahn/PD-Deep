# PD-Deep
Partial discharge detection, on medium voltage power lines, using deep learning. Project for CISC-867 at Queen's University, Kingston.

**Steps**
1. Download the data from kaggle.com, https://www.kaggle.com/c/vsb-power-line-fault-detection/data, and extract into data/raw/ folder
3. Run the data-prep if you want to do the data-prep yourself. Note, you will need > 45 GB ram.
4. Run the benchmark.py and primary.py files. You will have to create folders for logging and saving the checkpoint models.
5. Select the model with for the early stopping criteria; that is, highest accuracy, but stop at lowest validation loss.
6. Test the model in 3.0-test.ipynb. Note, there are already trained models that can be loaded, and are already configured in the 3.0-test.ipynb file.