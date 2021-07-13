import  numpy as np

INPUT_DIM = 4
OUT_DIM = 3
H_DIM = 5

X=np.random.randn(INPUT_DIM)

W1 = np.random.randn(INPUT_DIM, H_DIM)
b1 = np.random.randn(H_DIM)
W2 = np.random.randn(H_DIM, OUT_DIM)
b2 = np.random.randn(OUT_DIM)

def relu(t):
    return  np.maximum(t, 0)
def softmax(t):
    out = np.exp(t)
    return out/np.sum(out)

def predict(x):
    t1 = X @ W1 +b1
    h1 = relu(t1)
    t2= h1 @ W2 + b2
    z = softmax(t2)
    return z


probs = predict(X)
pred_class = np.argmax(probs)
class_name = ['Setosa', 'Versicolor', 'Virginica']
print('Predicted class:', class_name[pred_class])