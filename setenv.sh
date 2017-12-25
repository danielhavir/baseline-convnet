# Create directory for data
mkdir -p data

# Download data
datadir="./data/cifar-10-batches-py"
if [ ! -d "$datadir" ]
then
	tput setaf 3; echo "Downloading CIFAR dataset"; tput sgr0;
	file="cifar-10-python.tar.gz"
	wget "http://www.cs.toronto.edu/~kriz/$file" -O "./data/$file"
	tar -zxvf "./data/$file" -C "./data/"
else
	tput setaf 6; echo "CIFAR dataset already downloaded"; tput sgr0;
fi

# Create directory for TensorFlow checkpoints
mkdir -p tensorflow_model/checkpoints

# Create directory for TensorBoard logs
mkdir -p tensorflow_model/logs

# Create directory for PyTorch checkpoints
mkdir -p pytorch_model/checkpoints

tput setaf 2; echo "##### Environment is ready! #####"; tput sgr0;
