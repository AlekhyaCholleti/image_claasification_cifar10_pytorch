import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='/home/welcome/Downloads/ml_ai_dl', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='/home/welcome/Downloads/ml_ai_dl', train=False, download=True, transform=transform)

def get_train_loader(batchsize):
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True, num_workers=2)
	return train_loader


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CNNModel(torch.nn.Module):
	
	# shape of input x is (batchsize, 3, 32, 32)
	
	def __init__(self):
		super(CNNModel, self).__init__()
		
		#Input channels = 3, output channels = 6
		self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=5, stride=1)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
		
		self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5, stride=1)
		#400 input features, 120 output features (see sizing flow below)
		self.drop = torch.nn.Dropout(p=0.3)
		self.fc1 = torch.nn.Linear(16*5*5, 120)
		self.fc2 = torch.nn.Linear(120, 84)
		#64 input features, 10 output features for our 10 defined classes
		self.fc3 = torch.nn.Linear(84, 10)
		
	def forward(self, x):
		#Size changes from (batchsize, 3, 32, 32) to (batchsize, 6, 28, 28)
		x = self.conv1(x)
		x = F.relu(x)
		#Size changes from (batchsize, 6, 28, 28) to (batchsize, 6, 14, 14)
		x = self.pool(x)
		#Size changes from (batchsize, 6, 28, 28) to (batchsize, 16, 10, 10)
		x = self.conv2(x)
		x = F.relu(x)
		#Size changes from (batchsize, 16, 10, 10) to (batchsize, 16, 10, 10)
		x = self.pool(x)
		#flattening the input from (batchsize, 16, 5, 5) to (batchsize, *)
		x = x.view(x.shape[0], -1)
		#Size changes from (batchsize, 400) to (batchsize, 120)
		#x = self.drop(x)
		x = F.relu(self.fc1(x))
		#Size changes from (batchsize, 120) to (batchsize, 84)
		#x = self.drop(x)
		x = F.relu(self.fc2(x))
		#size changes from (batchsize, 84) to (batchsize, 10)
		#x = self.drop(x)
		x = self.fc3(x)
		return(x)

def createLossAndOptimizer(net, learning_rate=0.001):
	
	#Loss function
	loss = torch.nn.CrossEntropyLoss()
	
	#Optimizer
	optimizer = optim.Adam(net.parameters(), lr=learning_rate)
	
	return(loss, optimizer)


def trainNet(model, batch_size, n_epochs, learning_rate):
	#Get training data
	train_loader = get_train_loader(batch_size)
	n_batches = len(train_loader)
	
	#Create our loss and optimizer functions
	loss, optimizer = createLossAndOptimizer(model, learning_rate)
	
	#Loop for n_epochs
	for epoch in range(n_epochs):
		
		for i, data in enumerate(train_loader, 0):
			
			#Get inputs
			images, labels = data
			
			#Wrap them in a Variable object
			images, labels = Variable(images), Variable(labels)
			
			#Set the parameter gradients to zero
			optimizer.zero_grad()

			#Forward propagation 
			outputs = model(images)      
			
			#Calculating loss with softmax to obtain cross entropy loss
			loss_size = loss(outputs, labels)
		
			#Backward propation
			loss_size.backward()
		
			#Updating gradients
			optimizer.step()        
			
			#Total number of labels
			total = labels.size(0)
		
			#Obtaining predictions from max value
			_, predicted = torch.max(outputs.data, 1)
		
			#Calculate the number of correct answers
			correct = (predicted == labels).sum().item()
		
			if (i + 1) % 100 == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
				  		.format(epoch + 1, n_epochs, i + 1, len(train_loader), loss_size.item(),
						  		(correct / total) * 100))
		
	print("Training finished")

	torch.save(model.state_dict(), '/home/welcome/python_files/project/cifarmodel.ckpt')


CNN = CNNModel()
trainNet(CNN, batch_size=32, n_epochs=20, learning_rate=0.001)   


def testing(batchsize):
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchsize, shuffle=False, num_workers=2)

	testing_model = CNNModel()
	testing_model.load_state_dict(torch.load('/home/welcome/python_files/project/cifarmodel.ckpt'))

	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in test_loader:
			images, labels = Variable(images), Variable(labels)
			outputs = testing_model(images)
			total += labels.size(0)
			_, predicted = torch.max(outputs.data, 1)
			correct += (predicted == labels).sum().item()
		print(' Accuracy: {:.2f}%'.format((correct / total) * 100))	

testing(50)
