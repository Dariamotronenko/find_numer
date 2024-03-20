import numpy as np
import matplotlib.pyplot as plt

import utils

images, labels = utils.load.dataset()

weight_input_to_hidden = np.random.uniform(-0.5, 0.5, (20,784))
weight_hidden_to_output = np.random.uniform(-0.5, 0.5, (10,20))

bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

def load_dataset():
    with np.load('mnist.npz') as f:

        x_train = f['x_train'].astype('float32') / 255
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
        y_train = f['y_train']
        y_train = np.eye(10)[y_train]
        return x_train, y_train

epochs = 3
e_loss = 0
e_correct = 0
learning_rate = 0.01
    
for epoch in range(epochs):
    print(f'Epoch №{epoch}')
    

    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1,1))
        #вот тут опасность переобучения

        hidden_raw = bias_input_to_hidden + weight_input_to_hidden @ image
        hidden = 1 / (1 + np.exe(-hidden_raw)) #функция активации 

        output_raw = bias_hidden_to_output + weight_hidden_to_output @ hidden
        output = 1 / (1 + np.exe(-output_raw))

        #считаем потери и точность 
        e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(label))

        #Backpropagation algorim (справа идем налево)
        delta_output = output - label
        weight_hidden_to_output += - learning_rate + delta_output @ np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output


        delta_hidden = np.transpose(weight_hidden_to_output) @delta_output * (hidden * (1 - hidden))
        weight_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden

#ПРОВЕРЯЕМ
import random 
test_image = random.choise(images)

image = np.random(test_image, (-1, 1))


hidden_raw = bias_input_to_hidden + weight_input_to_hidden @ image
hidden = 1 / (1 + np.exe(-hidden_raw)) #функция активации 

output_raw = bias_hidden_to_output + weight_hidden_to_output @ hidden
output = 1 / (1 + np.exe(-output_raw))

plt.imshow(test_image.reshape(28, 28), cmap='Greys')
plt.title(f'NN suggests the number is: {output.argmax()}')
plt.show

