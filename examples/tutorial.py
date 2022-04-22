# This code belongs to the paper
#
# J. Hertrich and G. Steidl. 
# Inertial Stochastic PALM and Applications in Machine Learning. 
# Sampling Theory, Signal Processing, and Data Analysis, vol. 20, no. 4, 2022.
# https://doi.org/10.1007/s43670-022-00021-x
#
# Please cite the paper, if you use the code.

from palm_algs import *

# implement model functions

def H(X,batch):
    diffs1=tf.add(batch,-X[0])
    diffs2=tf.add(batch,-X[1])
    return tf.reduce_sum(diffs2**2)-tf.reduce_sum(diffs1**2)

def prox_1(x,lam):
    my_norm=tf.sqrt(tf.reduce_sum(x**2))
    if my_norm<1:
        return x
    return x/my_norm

def prox_2(x,lam):
    return tf.multiply(tf.math.sign(x),tf.math.maximum(tf.math.abs(x)-1/lam,0))

def f_1(x):
    my_norm=tf.math.sqrt(tf.reduce_sum(x**2))
    if my_norm>1.:
        a=tf.constant(np.inf,dtype=tf.float32)
    else:
        a=tf.constant(0,dtype=tf.float32)
    return a

def f_2(x):
    return tf.reduce_sum(tf.math.abs(x))
    
# initialization
d=5
inits=[np.zeros(d).astype(np.float32),np.zeros(d).astype(np.float32)]

n=10000
data=np.random.normal(loc=0.5,size=[n,d]).astype(np.float32)

# model declaration
model=PALM_Model(inits)
model.H=H
model.prox_funs=[prox_1,prox_2]
model.f=[f_1,f_2]

# run algorithm
method='iSPALM-SARAH'
optim=PALM_Optimizer(model,data=data,method=method)
optim.optimize(EPOCHS=10)
# print result
print('X_1='+str(model.X[0].numpy())+', X_2='+str(model.X[1].numpy()))
