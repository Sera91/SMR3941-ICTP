How to use the programs:

1. to use the main.py program: 

python3 main.py arg1 arg2 arg3 

arg1 = critic learning rate 

arg2 = actor learning rate 

arg3 = directory to save plots and data of the rewards during the learning procedure, and to save the weights of the neural networks used to implement actor, critic1, critic2 and their target networks 

arg4 = number of episodes for the training

2. to use the test.py program : 

python3 test.py arg1 arg2 arg3 

arg1 = directory where the networks parameters to load the model are saved 

arg2 = integer number of episodes 

arg3 = 0 to disable the graphic rendering, 1 or any other integer to enable it 

