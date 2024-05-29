import torch
import torchvision
import torchvision.transforms as transforms
import os

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split

DATASET_FOLDER=os.environ['SCRATCH']+'/datasets'

def iid_split(dataset, world_size, seed):
	return random_split(dataset, [1/world_size for _ in range(world_size)], torch.Generator().manual_seed(seed))

def non_iid_quantity_split(dataset, world_size, seed):
	if world_size==1:
		return [dataset]
	else:
		split=[]
		for rank in range (world_size):
			if rank==0:
				split.append(0.8)
			else:
				split.append(0.2/(world_size-1))
		return random_split(dataset, split, torch.Generator().manual_seed(seed))

def no_split(trainset, world_size):
	return [trainset for _ in range (world_size)]

def get_cifar10_train_loader(batch_size=4, train_samples=-1, rank=0, world_size=1, seed=42):
	transform = transforms.Compose(
		[transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root=DATASET_FOLDER, train=True, transform=transform)
	if train_samples > 0:
		trainset = torch.utils.data.Subset(trainset, list(range(0, train_samples)))

	#train_sampler = DistributedSampler(dataset=trainset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True)

	trainset_splits = iid_split(trainset, world_size, seed)
	#trainset_splits = non_iid_quantity_split(trainset, world_size, seed)
	#trainset_splits = no_split(trainset, world_size)

	loaders = []
	for split in trainset_splits:
		loaders.append(torch.utils.data.DataLoader(split, batch_size=batch_size, num_workers=0, pin_memory=True))

	return loaders

def get_cifar10_test_loader(batch_size=4, test_samples=-1):
	transform = transforms.Compose(
		[transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	testset = torchvision.datasets.CIFAR10(root=DATASET_FOLDER, train=False, transform=transform)

	if test_samples > 0:
		testset = torch.utils.data.Subset(testset, list(range(0, test_samples)))

	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=0, pin_memory=True)

	return testloader

# To be run on a computing node connected to the Internet
if __name__ == "__main__":
	torchvision.datasets.CIFAR10(root=DATASET_FOLDER, train=True, download=True)
	#torchvision.datasets.CIFAR100(root=os.environ['SCRATCH']+'/datasets', train=True, download=True)
