setup:
	pip install -r requirements.txt
	mkdir -p output images data
train-mnist-default:
	python training/ATACGAN_MNIST.py --verbose=False --t_model=LeNet5 --d_model=Discriminator_Combined_4Ca --g_model=Generator_3Ca
test-mnist-default:
	python testing/Test_MNIST_Model_Attack.py --name='$(name)' --test_size=500
