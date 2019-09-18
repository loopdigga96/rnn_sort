

from model import *




import sys
import itertools
import numpy as np  # Matrix and vector computation package
import matplotlib
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Fancier plots
from sklearn.metrics import accuracy_score, classification_report

# Set seaborn plotting style
sns.set_style('darkgrid')
# Set the seed for reproducability
np.random.seed(seed=1)


# In[3]:


def create_sort_dataset(dataset_length, seq_length, max_number=999):
    x_train = np.random.randint(low=0, high=max_number+1, size=(dataset_length, seq_length, 1))
    y_train = np.sort(x_train, axis=1)
    
    x_test = np.random.randint(low=0, high=max_number+1, size=(dataset_length, seq_length, 1))
    y_test = np.sort(x_test, axis=1)
    
    return x_train, y_train, x_test, y_test

def create_dummy_dataset(dataset_length, seq_length, max_number):
    lower_bound = -1 * max_number
    x_train = np.random.randint(low=lower_bound, high=max_number+1, size=(dataset_length, seq_length, 1))
    y_train = np.where(x_train.sum(axis=2) > 0, 1, 0).reshape(x_train.shape)
    
    x_test = np.random.randint(low=lower_bound, high=max_number+1, size=(dataset_length, seq_length, 1))
    y_test = np.where(x_test.sum(axis=2) > 0, 1, 0).reshape(x_train.shape)
    
    return x_train, y_train, x_test, y_test
    


# In[4]:


if __name__ == "__main__":
    # Set hyper-parameters
    batch_size = 20  # Size of the minibatches (number of samples)
    max_num = 8
    seq_length = 9
    hidden_size = 30
    dataset_size = 200
    epoch = 50

    x_train, y_train, x_test, y_test = create_sort_dataset(dataset_size, seq_length, max_num)

    input_size = 1
    output_size = max_num+1

    # Create the network
    model = ModelSort(input_size, output_size, hidden_size, seq_length)

    # Create a list of minibatch losses to be plotted
    ls_of_loss = [
        model.loss(model.predict_proba(x_train[:batch_size]), y_train[:batch_size])]

    # Iterate over some iterations
    for i in range(epoch):
        print(f'Epoch {i+1}/{epoch}')
        # Iterate over all the minibatches
        for mb in range(dataset_size // batch_size):
            x_batch = x_train[mb:mb + batch_size]  # Input minibatch
            y_batch = y_train[mb:mb + batch_size]  # Target minibatch
            model.train_on_batch(x_batch, y_batch)
            # linear_out, rnn_states, out, probabilities = model.forward(x_batch)
            # model.backward(x_batch, probabilities, linear_out, rnn_states, y_batch)

            ls_of_loss.append(model.loss(model.predict_proba(x_batch), y_batch))


            # V_tmp = [v * momentum_term for v in Vs]
            # Update each parameters according to previous gradient
            # for pIdx, P in enumerate(model.get_params_iter()):
            #     P += V_tmp[pIdx]
            # Get gradients after following old velocity
            # Get the parameter gradients
            # backprop_grads = model.getParamGrads(x_batch, y_batch)
            # # Update each parameter seperately
            # for pIdx, P in enumerate(model.get_params_iter()):
            #     P -= learning_rate * backprop_grads[pIdx]
                # Update the Rmsprop moving averages
                # maSquare[pIdx] = lmbd * maSquare[pIdx] + (
                #     1-lmbd) * backprop_grads[pIdx]**2
                # # Calculate the Rmsprop normalised gradient
                # pGradNorm = ((
                #     learning_rate * backprop_grads[pIdx]) / np.sqrt(
                #     maSquare[pIdx]) + eps)
                # # Update the momentum
                # Vs[pIdx] = V_tmp[pIdx] - pGradNorm
                # P -= pGradNorm   # Update the parameter
            # Add loss to list to plot




        print("Cross entropy loss: ", np.mean(ls_of_loss))






