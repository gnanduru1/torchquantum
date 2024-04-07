"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ClassicalNN(nn.Module):
    def __init__(self, n_classes):
        super(ClassicalNN, self).__init__()
        #TODO change default linear layer sizes to match kind of what QNN does
        self.fc1 = nn.Linear(784, 512) # First hidden layer
        self.fc2 = nn.Linear(512, 256) # Second hidden layer
        self.fc3 = nn.Linear(256, n_classes) # Output layer

    def forward(self, x):
        print("x.shape", x.shape)
        x = F.relu(self.fc1(x)) # Activation function for first hidden layer
        x = F.relu(self.fc2(x)) # Activation function for second hidden layer
        x = self.fc3(x) # No activation, raw scores
        return F.log_softmax(x, dim=1) # Log-softmax for output