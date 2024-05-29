import torch
import copy

from tqdm import tqdm

def train(model, criterion, optimizer, train_loader, epochs, device):
	model.train()
	for epoch in range(epochs):  # loop over the dataset multiple times
		running_loss = 0.0
		pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=len(train_loader), dynamic_ncols=True)
		for i, data in enumerate(train_loader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data[0].to(device), data[1].to(device)
			# forward + backward + optimize
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			# zero the parameter gradients
			optimizer.zero_grad()

			pbar.update(1)
			pbar.set_description(f"Training Epoch: {epoch+1}/{epochs}, step {i}/{len(train_loader)} completed (loss: {loss.detach().float():.8f})")

	print('Finished Training')
	return model

def test(model, test_loader, device):
	model.eval()
	correct = 0
	total = 0
	# since we're not training, we don't need to calculate the gradients for our outputs
	with torch.no_grad():
		for data in tqdm(test_loader, colour="green", desc="Test Epoch", dynamic_ncols=True):
			inputs, labels = data[0].to(device), data[1].to(device)
			# calculate outputs by running images through the network
			outputs = model(inputs)
			# the class with the highest energy is what we choose as prediction
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print(f'Accuracy of the network on the test images: {100 * correct // total} %')

# TRY TO IMPLEMENT THIS FUNCTION
# The Federated Average algorithm takes as imput N models and returns the average of their weights
# In this case we take as input a list of PyTorch models, and return the aggregated state dict 
def aggregate(models):

	#Remove these two lines
	print("Aggregation function missing! Implement it :)")
	quit()
	# Collect models' state dicts
	# average the state dicts

	return #aggregated state dict
