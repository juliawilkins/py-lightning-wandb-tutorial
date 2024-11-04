# 🐱‍💻 MARL ML Tutorial: ⚡ PyTorch Lightning / Weights & Biases (WandB) 📜

>  **[Tutorial Page](https://julianotes.notion.site/MARL-ML-Tutorial-PyTorch-Lightning-Weights-Biases-WandB-Tutorial-12d06e9a3217809490cfdcad3ad48614?pvs=4)**  
> [Julia Wilkins](https://juliawilkins.github.io), [Xavier Juanola](https://xavijuanola.github.io/)

**Abstract**: This tutorial by Julia Wilkins and Xavier Juanola provides an introduction to PyTorch Lightning and Weights & Biases, showcasing how to simplify deep learning workflows. Participants will learn to integrate these tools into existing projects, enhancing reproducibility and scalability while enjoying built-in best practices like automated logging and advanced experiment tracking. The session includes lab time for hands-on experience in adapting PyTorch code to leverage these powerful tools, with additional resources for setting up environments and downloading necessary datasets.

**Keywords**: PyTorch Lightning, Weights & Biases

*****************

## Index
- [Why PyTorch Lightning](#why-pytorch-lightning)
- [Why Weights & Biases](#why-weights--biases)
- [Installation and Getting Started](#installation-and-getting-started)
- [Lab Template Project](#lab-template-project)
- [References](#references)


*****************

## Why PyTorch Lightning?

[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) is an advanced framework that builds upon PyTorch, aiming to simplify the deep learning and model development process.

**Reasons to use PyTorch Lightning**

- *Simplifies code* ➡️ Reduces the amount of code you write for training loops, device management (*which GPU to use*), etc.
- *Consistent setup* ➡️ Setup is the same *(almost always)* aiding reproducibility.
- **Easy Scalability** ➡️ Handles complex hardware scaling internally with minimal code adjustments!!
- Built-in **best practices** ➡️ Integrates essential practices such as logging, checkpointing, early-stopping, learning-rate scheduling, easy fine-tuning, etc.
- Trade-off between easy development ↔️ Hardcore abstraction.
- **Integration with ML Tools**: Weights & Biases, Hydra, and more.

Need more reasons? Take a look at [Pytorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)

*****************

## Why Weights & Biases?

[Weights & Biases](https://wandb.ai/site/) is a powerful tool designed to help machine learning practitioners track their experiments, visualize data, and optimize models more effectively.

**Reasons to use Weights & Biases**

- **Easy integration** ➡️ Easy integration with PyTorch, TensorFlow and Keras, as well as with other tools like Jupyter Notebooks (Minimal code changing)
- **Comprehensive Experiment Tracking** ➡️ Keep detailed logs of every experiment including code version, metrics, hyperparameters, output files, and automatically organizes your experiment history making it easy to compare and reproduce results
- **Rich Visualizations** ➡️ Generate rich visual reports, including plots, images, audios, etc.
- **Real-time Monitoring** ➡️ View live updates of your training and validation metrics
- **Artifact tracking** ➡️ Version and track models and other files as part of your pipeline

*****************

## Installation and Getting Started

Clone our repo to get started with this tutorial:

```bash
git clone https://github.com/juliawilkins/py-lightning-wandb-tutorial
```

The repo contains:

- `data.py`: This file contains the ESC50 dataloader. The dataloader returns batches of spectrograms and target labels 0-49.
- `model.py`: Contains the main model code, the class `SimpleCNN`.
- `train.py`: This file contains the vanilla PyTorch code for training the sound classification model, with CLI logging and CSV exporting for metrics.
- `train_wandb.py`: The same model code as in `train.py`, but with logging using Weights & Biases.
- `train_lightning.py`: This is the PyTorch Lightning version of `train.py`, that also uses Weights & Biases for logging.
- `train_lightning_template.py`: This is the template version of - `train_lightning.py`. The methods are the same, but you’ll use this to learn how to port your code from vanilla PyTorch to Lightning.
-  `requirements.txt`: Contains all the necessary packages for this project.

### Set up a new virtual environment

Let's use Conda and pip for this example:

```bash
conda create -n esc50env python=3.9
conda activate esc50env
pip install -r requirements.txt
```

### Other important TODOs before moving on

- [ ]  Data: We will be working with the ESC-50 environmental sound data. There are 2000 5-second audio samples across 50 labeled classes. Download the dataset from this link: ESC-50 Download. Unzip ESC-50-MASTER.zip and move this into the base directory of the tutorial repo (i.e., the data top-level folder should be at the same level as data.py).
- [ ]  Set up a free account on Weights & Biases: Go to Weights & Biases and create an account. You will get an API key that you’ll need to hold on to to log-in when you try to instantiate your WandB run for the first time.

### Verify that everything is working

Confirm that everything is set up correctly by running:

```bash
python train.py
python train_lightning.py
```

These commands should run without errors, confirming that your environment is correctly configured with PyTorch, Lightning, and WandB.

YAY! 😊 You’re ready to level up….

*****************

## Lab Template Project

We’ve created a template version of the pytorch lightning-adapted code for `train.py` in **`train_lightning_template.py`**. Spend the next 20 minutes or so filling out the template, porting the code from `train.py` into the lightning format and also playing with some weights and biases logging. Remember that you can look at the “complete” version in `train_lighting.py`, but try to work on the template yourself first!

**Extra credit ideas:**

- [ ]  Add more custom training parameters in `Trainer()`, playing with things such as `limit_val_batches`, `accelerator`, other `callbacks` such as `EarlyStopping` etc. there’s a lot you can do here! Explore [the docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html).
- [ ]  Try out logging a [matplotlib plot to wandb](https://docs.wandb.ai/guides/track/log/plots/)! Maybe you could plot the per-class accuracy on a bar chart (though this would be a lot of bars…)

*****************

## References

- [Lightning website](https://lightning.ai/docs/pytorch/stable/)
- [Weights & Biases](https://wandb.ai/site/)
