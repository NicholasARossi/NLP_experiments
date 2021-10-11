import numpy as np
from absl import flags

flags.DEFINE_float('learning_rate',0.01,'')
flags.DEFINE_integer('n_iters',10,'')
FLAGS = flags.FLAGS


X = np.array([1,2,3,4])
y = np.array([2,4,6,8])

w = 0.0

def forward(X):
    return X*w

def loss(y,y_predicted):

    return ((y-y_predicted)**2).mean()


# Loss = 1/N * (Xw - y)**2
# Loss = 1/N * (X**2 - 2xy - y**2)
# dLoss/dw = 1/N * (2x - 2y)

def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

print(f'Prediction before Training : f(5) = {forward(5):.3f}')

for epoch in range(FLAGS.n_iters):
    y_pred = forward(X)

    l = loss(y,y_pred)

    dw = gradient(X,y,y_pred)

    w -= dw *FLAGS.learning_rate

    if epoch % 1 == 0:
        print(f'epoch {epoch +1} : w = {w:.3f} , loss = {l:.3f}')

print(f'Prediction after Training : f(5) = {forward(5):.3f}')
