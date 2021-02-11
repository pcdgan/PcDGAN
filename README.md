# PcDGAN: A Continuous Conditional Diverse Generative Adversarial Network For Inverse Design
![PcDGAN Architecture](https://drive.google.com/uc?export=view&id=1iFZCPvztdxz9jFEXR-Y1nVp_JHhFvBPv)
This code can be used to reproduce the results presented in the paper. Citation information and paper information will be made available here once the paper is approved.

## Required packages

- tensorflow > 2.0.0
- sklearn
- numpy
- matplotlib
- seaborn
- tensorflow_addons
- tensorflow_probability
- tqdm
- tabulate

## Usage

### Synthetic examples

1. Go to example directory:

   ```bash
   cd Synthetic
   ```

2. Train Models:

   ```bash
   python train_models.py
   ```

   positional arguments:
    
   ```
   model PcDGAN or CcGAN
   dataset	dataset name (available datasets are Uniform, Uneven, Donut2D)
   ```

   optional arguments:

   ```
   -h, --help   show this help message and exit
   --dominant_mode DOMINANT_MODE    The dominant mode for uneven dataset. Default: 1, Options: Any integet between 0 and 5
   --mode MODE  Mode of operation, either train or evaluate. Default: Train
   --vicinal_type   The type of vicinal approach. Default: soft, Options: soft, hard
   --kappa    Vicinal loss kappa. If negative automatically calculated and scaled by the absolute value of the number. Default: -1.0 for PcDGAN -2.0 for CcGAN
    --sigma   Vicinal loss sigma. If negative automatically calculated. Default: -1.0
    --lambda0   PcDGAN lambda0. Default: 3.0
    --lambda1   PcDGAN lambda1. Default: 0.5
    --lambert_cutoff    PcDGAN parameter "a". Default: 4.7
    --gen_lr    Generator learning rate. Default: 1e-4
    --disc_lr   Discriminator learning rate. Default: 1e-4
    --train_steps   Number of training steps. Default: 50000
    --batch_size    GAN training Batch size. Default: 32
    --id    experiment ID or name. Default:
    --size  Number of samples to generate at each step for evaluation. Default: 1000
   ```

   The trained models will be saved under the specified dataset folder under the subdirectory of Weights under the each models folder and the result plots will be saved under the directory of the dataset under the subdirectory Evaluation under each models folder. Change the id argument everytime to prevent over-writing previous weights.
   
   Note that we can set `lambda0` and `lambda1` to zeros to train a CcGAN with only singular vicinal loss.

3. To reproduce the results of the paper train atleast 3 versions of each model(although for the paper 10 were trained for each model) by changing the id argument during training and run the following:

    ```bash
    python evaluation.py
    ```

    positional arguments:
    
   ```
   dataset	dataset name (available datasets are Uniform, Uneven, Donut2D)
   ```

   optional arguments:


   ```
   -h, --help   show this help message and exit
   --size SIZE  Number of samples to generate at each step for evaluation. Default: 1000
   ```
   
   After this the Resulting Figures(Similar to what is presented in the paper) will be produced under the dataset directory under the subdirectory of Evaluation.

### Airfoil example

1. Install [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/). Only necessary if you are on Linux. For windows the executables are provided under the XFOIL_Windows folder.

2. Go to example directory:

   ```bash
   cd Airfoil
   ```

3. Download the airfoil dataset [here](https://drive.google.com/drive/folders/1x1SrAX28ajLD0T_zbTUhcYxg2M5kudHm?usp=sharing) and extract the NPY files into `Airfoil/data/`.


4. First train an embedder estimator pair:

   ```bash
   python train_estimator_embedder.py
   ```
    optional arguments:

   ```
   -h, --help   show this help message and exit
   --data   The path to the data. Default: ./data
   --estimator_save_name    The file name of the best checkpoint saved in the weights estimator folder. Default:best_checkpoint
   --embedder_save_name     The file name of the best checkpoint saved in the embedder weights folder. Default:best_checkpoint
   --estimator_lr   Initial estimator learning rate before decay. Default: 1e-4
   --embedder_lr    Initial embedder learning rate before decay. Default: 1e-4
   --estimator_train_steps  Number of training steps for estimator. Default: 10000
   --embedder_train_steps   Number of training steps for embedder. Default: 10000
   --estimator_batch_size   Batch size for estimator Default: 256
   --embedder_batch_size    Batch size for embedder Default: 256
   ```

   The weights of both models will be saved under the Weights folder. Remember if you name the pair differently for GAN training. Also use only one pair for each experiment as cross-validation is done using one pair which both models were based on.

5. Train Models:

   ```bash
   python train_models.py
   ```

   positional arguments:
    
   ```
   model PcDGAN or CcGAN
   ```

   optional arguments:

   ```
   -h, --help   show this help message and exit
   --mode MODE  Mode of operation, either train or evaluate. Default: Train
   --vicinal_type   The type of vicinal approach. Default: soft, Options: soft, hard
   --kappa  Vicinal loss kappa. If negative automatically calculated and scaled by the absolute value of the number. Default: -1.0 for PcDGAN -2.0 for CcGAN
   --sigma  Vicinal loss sigma. If negative automatically calculated. Default: -1.0
   --estimator  Name of the estimator checkpoint saved in the weights folder. Default: best_checkpoint
   --embedder   Name of the embedder checkpoint saved in the weights folder. Default: best_checkpoint
   --lambda0    PcDGAN lambda0. Default: 3.0
   --lambda1    PcDGAN lambda1. Default: 0.4
   --lambert_cutoff PcDGAN parameter "a". Default: 4.7
   --gen_lr GEN_LR  Generator learning rate. Default: 1e-4
   --disc_lr DISC_LR    Discriminator learning rate. Default: 1e-4
   --train_steps    Number of training steps. Default: 20000
   --batch_size     GAN training Batch size. Default: 32
   --size   Number of samples to generate at each step. Default: 1000
   --id     experiment ID or name. Default:
   ```
   Change the id argument everytime to prevent over-writing previous weights. Train each model atleast 3 times to reproduce paper results (for the paper 10 models were trained). The results of each model will be saved under the Evaluation Directory.


6. To reproduce the results of the paper train atleast 3 versions of each model(although for the paper 10 were trained for each model) by changing the id argument during training and run the following:

    ```bash
    python evaluation.py
    ```

   optional arguments:


   ```
   -h, --help   show this help message and exit
   --estimator  Name of the estimator checkpoint saved in the weights folder. Default: best_checkpoint
   --embedder   Name of the embedder checkpoint saved in the weights folder. Default: best_checkpoint
   --simonly    Only evaluate based on simluation. Default: 0(False). Set to 1 for True
   --size   Number of samples to generate at each step. Default: 1000
   ```
   
   After this the Resulting Figures(Similar to what is presented in the paper) will be produced under the Evaluation directory.