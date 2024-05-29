import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
import utils
import copy

from torchvision.models import resnet18
from datasets import get_cifar10_train_loader, get_cifar10_test_loader
from models import Net


def main(args, device):
	federation_client = [None for _ in range(args.clients)]
	federation_train_loader = [None for _ in range(args.clients)]
	federation_criterion = [None for _ in range(args.clients)]
	federation_optimizer = [None for _ in range(args.clients)]

	federation_test_loader = get_cifar10_test_loader(batch_size=args.batch_size, test_samples=args.test_samples)

	# Initialisation
	for rank in range(args.clients):
		federation_client[rank] = resnet18().to(device) #resnet18() # otherwise Net()

		# Loss & optimizer
		federation_criterion[rank] = nn.CrossEntropyLoss() # otherwise MSE
		federation_optimizer[rank] = optim.SGD(federation_client[rank].parameters(), lr=args.lr, momentum=args.momentum) # otherwise Adam, AdamW
	
	# Dataset loading
	federation_train_loaders = get_cifar10_train_loader(batch_size=args.batch_size, train_samples=args.train_samples, rank=rank, world_size=args.clients, seed=args.seed) # Otherwise cifar100

	# Federated Learning
	print(f"Testing the initial model...")
	utils.test(federation_client[0], federation_test_loader, device)
	for fl_round in range(args.rounds):
		print(f"Federated round {fl_round}")
		for rank in range(args.clients):
			print(f"Training model {rank}...")
			federation_client[rank]=utils.train(federation_client[rank], federation_criterion[rank], federation_optimizer[rank], federation_train_loaders[rank], args.epochs, device)
			utils.test(federation_client[rank], federation_test_loader, device)
			
		aggregated_state_dict = utils.aggregate(federation_client)
		for rank in range(args.clients):
			federation_client[rank].load_state_dict(aggregated_state_dict)
		print(f"Testing the aggregated model...")
		utils.test(federation_client[0], federation_test_loader, device)

start_time = time.time()
if __name__ == "__main__":
	# Command line arguments parsing
	parser = argparse.ArgumentParser(description="Federated Learning tutorial - ICTP ASAML")
	parser.add_argument("--clients", default=2, type=int, help="Number of clients participating to the federation")
	parser.add_argument("--rounds", default=1, type=int, help="Number of federated rounds")
	parser.add_argument("--epochs", default=1, type=int, help="Number of local epochs")
	parser.add_argument("--batch-size", default=4, type=int, help="Batch size")
	parser.add_argument("--train-samples", default=-1, type=int, help="Train samples to load")
	parser.add_argument("--test-samples", default=-1, type=int, help="Test samples to load")
	parser.add_argument("--lr", default=0.001, type=int, help="Learning rate")
	parser.add_argument("--momentum", default=0.9, type=int, help="momentum")
	parser.add_argument("--seed", type=int, default=42, help="Data partitioning seed")
	parser.add_argument("--cpu", action='store_true', help="Force CPU usage")
	args = parser.parse_args()

	print(f"*** FL simulation setup: clients->{args.clients}, rounds->{args.clients}, seed->{args.seed} epochs->{args.epochs} ***")

	if args.cpu:
		print("Forcing CPU usage")
		device=torch.device("cpu")
	else:
		# Verifying CUDA availability
		is_cuda_available=torch.cuda.is_available()
		device=torch.device(0 if torch.cuda.is_available() else "cpu")
		torch.cuda.set_device(0)
		
		print(f"PyTorch CUDA availability: {torch.cuda.is_available()}")
		print(f"Currently used device: {device}")
		if is_cuda_available:
			print(f"Number of available GPUs: {torch.cuda.device_count()}")	
			print(f"Device type: {torch.cuda.get_device_name(device)}")
		
	# Simulation starts
	main(args, device)
print(f"--- FL simulation elapsed time: %s seconds ---" % (time.time() - start_time))
