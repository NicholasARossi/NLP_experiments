import torch

from absl import flags

flags.DEFINE_float('learning_rate',0.01,'')
flags.DEFINE_integer('n_iters',10,'')
FLAGS = flags.FLAGS

N_ITERS = 20
LEARNING_RATE = 0.01

X = torch.tensor([1,2,3,4],dtype=torch.float32)
y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0,dtype=torch.float32, requires_grad=True)


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

for epoch in range(N_ITERS):
    y_pred = forward(X)

    l = loss(y,y_pred)

    l.backward()
    with torch.no_grad():
        w -= w.grad *LEARNING_RATE


    w.grad.zero_()
    if epoch % 1 == 0:
        print(f'epoch {epoch +1} : w = {w:.3f} , loss = {l:.3f}')

print(f'Prediction after Training : f(5) = {forward(5):.3f}')
