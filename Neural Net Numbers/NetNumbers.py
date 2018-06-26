import numpy
# sigmoid func
import scipy.special
# plotting arrays
import matplotlib.pyplot
import scipy.misc
import glob

# neural network class
class neuralNetwork:
    
    # initialize
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        
        # set nodes in each layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # set learning rate
        self.lr = learningrate
        
        # link weight matrices, input to hidden and hidden to output
        # weights are w_i_j, link node i to node j in next layer
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        # activation function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    # train network
    def train(self, inputs_list, targets_list):
         # convert inputs into 2D array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate input of hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate emerging signals
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate input of final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate signals emerging from final layer
        final_outputs = self.activation_function(final_inputs)
        
        # error, (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error, split by weight
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # update hidden to output weights
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # update input to hidden weights
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    
    # query network
    def query(self, inputs_list):
        # convert inputs into 2D array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate input of hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate emerging signals
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate input of final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate signals emerging from final layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


# input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
    
# learning rate
learning_rate = 0.1
    
# create instance
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
# load mnist training data
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# epochs is the number of loops of the training data set
epochs = 5
    
for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split by ,
        all_values = record.split(',')
        # scale and shift inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create target output
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# load the mnist test data CSV file into a list
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network
# scorecard for how well the network performs, initially empty
scorecard = []
# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print(label, "network's answer")
    # append correct or incorrect to list
    
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to
        scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to
        scorecard
        scorecard.append(0)
        pass
    pass

print(scorecard)
# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() /
scorecard_array.size)

# our own image test data set
our_own_dataset = []

# load the png image data as test data set
for image_file_name in glob.glob('my_own_images/my_own_?.png'):
    
    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])
    
    # load image data from png files into an array
    print ("loading ... ", image_file_name)
    img_array = scipy.misc.imread(image_file_name, flatten=True)
    
    # reshape from 28x28 to list of 784 values, invert values
    img_data  = 255.0 - img_array.reshape(784)
    
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))
    
    # append label and image data  to test data set
    record = numpy.append(label,img_data)
    our_own_dataset.append(record)
    
    pass

# test the neural network with our own images

# record to test
item = 9

# plot image
matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')

# correct answer is first value
correct_label = our_own_dataset[item][0]
# data is remaining values
inputs = our_own_dataset[item][1:]

# query the network
outputs = n.query(inputs)
print (outputs)

# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)
print("network says ", label)
# append correct or incorrect to list
if (label == correct_label):
    print ("match!")
else:
    print ("no match!")
    pass
