# -*- coding: utf-8 -*-
"""
@author: Ahmad Qadri
Function approximation using Backpropagation 
Using PyTorch package
Fitting a 4th order polynomial

"""

import csv
import matplotlib.pyplot as plt
import torch
import numpy as np


def csvScatterPlot(x, y1, y2, x_label, y_label, y1_label, y2_label, title, subtitle, output_folder, filename):      
    plt.scatter(x, y1, c="blue", alpha=0.5, marker='o', label=y1_label)
    
    if y2:
        plt.scatter(x, y2, c="red", alpha=1.0, marker='o', label=y2_label)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(subtitle)
    plt.suptitle(title, fontsize=15)

    plt.legend(loc='upper left')
    plt.savefig(output_folder+filename)
    plt.close()



if __name__ == "__main__":
    data, x_orig, y_orig = [], [], []
    epochs = 50000
    learning_rate = 1e-5
    
    
    #----  reading data
    with open('data\\data.csv', 'r') as file:
        reader = csv.reader(file)
        cols = next(reader, None)
        
        for row in reader:
            xy_pair = []
            
            for item in row:
                xy_pair.append(float(item))

            data.append(xy_pair)        
    
    print(f'> columns = {cols}')
    for xy_pair in data:
        x_orig.append(xy_pair[0])
        y_orig.append(xy_pair[1])
        
   
    #----  visualizing
    csvScatterPlot(x_orig, y_orig, [], cols[0], cols[1], 'data', '', 'Function Approximation', \
                   'using backpropagation (PyTorch)', 'output//', 'xy_scatterplot.png')


    #----  using PyTorch to approximate best fit to the data with 4th order polynomial
    #      y = b_0 + b_1(x) + b_2(x^2) + b_3(x^3) + b_4(x^4)

    x = torch.from_numpy(np.array(x_orig))
    y = torch.from_numpy(np.array(y_orig))
    
    # starting with randomly assigned coefficients for the polynomial
    b0 = torch.randn((1), device=torch.device("cpu"), dtype=torch.float, requires_grad=True)
    b1 = torch.randn((1), device=torch.device("cpu"), dtype=torch.float, requires_grad=True)
    b2 = torch.randn((1), device=torch.device("cpu"), dtype=torch.float, requires_grad=True)
    b3 = torch.randn((1), device=torch.device("cpu"), dtype=torch.float, requires_grad=True)
    b4 = torch.randn((1), device=torch.device("cpu"), dtype=torch.float, requires_grad=True)

    
    for epoch in range(epochs):
        
        # In each epoch, we will do a forward pass, compute the loss, backpropagate,
        # calculate gradients for the coefficients and update them. 
        
        # forward pass
        y_hat = b0 + b1*x + b2*x**2 + b3*x**3 + b4*x**4
        
        # compute loss
        loss = (y_hat - y).pow(2).sum()
        print(f'epoch {epoch}:  loss = {loss}')
        
        # backpropagate
        loss.backward()
        
        # get gradients, update weights (remember to zero out the gradients)
        with torch.no_grad():
            b0 -= learning_rate * b0.grad
            #b0.grad = None  or
            b0.grad.zero_()
            
            b1 -= learning_rate * b1.grad
            b1.grad.zero_()
            
            b2 -= learning_rate * b2.grad
            b2.grad.zero_()
            
            b3 -= learning_rate * b3.grad
            b3.grad.zero_()
            
            b4 -= learning_rate * b4.grad
            b4.grad.zero_()
            
        
    #---- get final coefficients for the polynomial and print
    b0 = round(b0.item(), 4)
    b1 = round(b1.item(), 4)
    b2 = round(b2.item(), 4)
    b3 = round(b3.item(), 4)
    b4 = round(b4.item(), 4)

    print(f'\nFunction: y = {b0} + {b1} x + {b2} x^2 + {b3} x^3 + {b4} x^4')


    #---- visualize the fit by overlaying yhat with y on the scatterplot
    y_hat = []
    
    for x in x_orig:
        y_hat.append(b0 + b1*x + b2*x**2 + b3*x**3 + b4*x**4)
    
    csvScatterPlot(x_orig, y_orig, y_hat, cols[0], cols[1], 'data', 'function', 'Function Approximation', \
                   'using backpropagation (PyTorch)', 'output//', 'func_scatterplot.png') 