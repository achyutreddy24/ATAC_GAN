# ATAC_GAN

## Report

[ATAC-GAN Report](https://drive.google.com/file/d/1FxP3I2eQDjPES9NP_Wlyvwer1MHmId_o/)

This is the most recent report on the progress of the ATAC-GAN research project by Calvin Hirsch and Achyut Reddy.

## Instructions

Install the python packages in requirements.txt in order to begin.

In order to train ATAC-GAN, run training/ATACGAN_MNIST.py. Use `python ATACGAN_MNIST.py -h` to see the parameters that can be set. The generator model, discriminator model, and target model all must be set as command line arguments. The options for these can be found in the corresponding file in the 'models' directory.
Sample command: `python training/ATACGAN_MNIST.py --t_model LeNet5 --g_model Generator_Cool --d_model Disc_Comb_Deotte`
