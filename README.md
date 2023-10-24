# Intelligent Module for System Trading of Financial Markets Assets Based on a Deep Neural Network and the DQN Learning Algorithm

**Authors:**  Kovaleva Maria, Kharitonov Alexander

This project uses DQN algorithm with two models (online and target) and a buffer to learn deep neural network to trade on the Moscow Exchange. 


## Data and Checkpoints
Raw and preprocessed data as well as weights for pretrained models available here: https://disk.yandex.ru/d/bHEs67EgCV2Gyw

## Code

* Gym like environment in [env.py](/env.py)
* Model in [models.py](/models.py)
* Trainer with training loop and logging in [trainer.py](/trainer.py)
* Setting hyperparameters and run training in [run_train.py](/run_train.py)
* Run evaluation in [run_test.py](/run_test.py)

## Reproducibility

* All necessary dependencies are specified in the `requirements.txt`.
* To train a new model you can use the following command: `python run_train.py`.
If you would like to adjust any hyperparameter you can pass it as a argument. For instance prompt `python run_train.py --epsilon 0.1` launchs the training process with probability of choosing random action equal to `0.1`.
* To evaluate a quality of the model and plot the test reward you can use the prompt `python run_test.py`

## Literature

The project is based on the article: https://ieeexplore.ieee.org/document/9681753
