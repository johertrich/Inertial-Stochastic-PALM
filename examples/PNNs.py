# This code belongs to the paper
#
# J. Hertrich and G. Steidl. 
# Inertial Stochastic PALM and Applications in Machine Learning. 
# Sampling Theory, Signal Processing, and Data Analysis, vol. 20, no. 4, 2022.
# https://doi.org/10.1007/s43670-022-00021-x
#
# Please cite the paper, if you use the code.

from palm_algs import *
import numpy.random
import numpy.matlib
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pickle
import os

#load and normalize data
mnist=tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=1.0*x_train
x_test=1.0*x_test
x_train_flat=[]
x_test_flat=[]
y_train_vec=[]
y_test_vec=[]

for i in range(0,len(x_train)):
    x_train_flat.append(x_train[i,:,:].reshape((28*28)))
    y_vec=np.zeros(10)
    y_vec[y_train[i]]=1.0
    y_train_vec.append(y_vec)

for i in range(0,len(x_test)):
    x_test_flat.append(x_test[i,:,:].reshape((28*28)))
    y_vec=np.zeros(10)
    y_vec[y_test[i]]=1.0
    y_test_vec.append(y_vec)


x_train=1.0*np.array(x_train_flat).astype(np.float32)
y_train=1.0*np.array(y_train_vec).astype(np.float32)
x_test=1.0*np.array(x_test_flat).astype(np.float32)
y_test=1.0*np.array(y_test_vec).astype(np.float32)

mean_x_train=1.0/len(x_train)*np.sum(x_train,axis=0)

x_train=x_train-np.matlib.repmat(mean_x_train,len(x_train),1)
x_test=x_test-np.matlib.repmat(mean_x_train,len(x_test),1)

max_x_train=np.max(np.abs(x_train))
x_train=x_train/max_x_train
x_test=x_test/max_x_train
print(np.prod(y_train.shape))

# parameters
n=x_train.shape[1]
sizes=[784,400,200]
my_activation=tf.keras.activations.elu

# required functions for PALM models
def H(X,batch):
    # computes the loss function of the model on the data contained in batch
    params=X[0]
    Ts=[]
    bs=[]
    ind=0
    for i in range(len(sizes)):
        Ts.append(tf.reshape(params[ind:(ind+sizes[i]*n)],[n,sizes[i]]))
        ind+=sizes[i]*n
        bs.append(params[ind:(ind+sizes[i])])
        ind+=sizes[i]
    Ts.append(tf.reshape(params[ind:],[10,n]))
    x=batch[:,:n]
    y=batch[:,n:]
    for i in range(len(sizes)):
        x=tf.linalg.matvec(Ts[i],x,transpose_a=True)+bs[i]
        x=my_activation(x)
        x=tf.linalg.matvec(Ts[i],x)
    x=tf.linalg.matvec(Ts[-1],x)
    x=tf.keras.activations.sigmoid(x)
    loss=tf.reduce_sum((x-y)**2)
    return loss

def accuracy(X,batch):
    # computes the accuracy of the model on the data contained in batch
    params=X[0]
    Ts=[]
    bs=[]
    ind=0
    for i in range(len(sizes)):
        Ts.append(tf.reshape(params[ind:(ind+sizes[i]*n)],[n,sizes[i]]))
        ind+=sizes[i]*n
        bs.append(params[ind:(ind+sizes[i])])
        ind+=sizes[i]
    Ts.append(tf.reshape(params[ind:],[10,n]))
    x=batch[:,:n]
    y=batch[:,n:]
    for i in range(len(sizes)):
        x=tf.linalg.matvec(Ts[i],x,transpose_a=True)+bs[i]
        x=my_activation(x)
        x=tf.linalg.matvec(Ts[i],x)
    x=tf.linalg.matvec(Ts[-1],x)
    x=tf.keras.activations.sigmoid(x)
    x_max=tf.argmax(x,axis=1)
    y_max=tf.argmax(y,axis=1)
    correct=tf.reduce_sum(tf.cast(tf.equal(x_max,y_max),dtype=tf.float32))
    accuracy=correct/y.shape[0]
    return accuracy.numpy()

@tf.function
def proj_orth(X):
    # Projects the matrix X onto the Stiefel manifold.
    num_iter=4
    Y=X
    for i in range(num_iter):
        Y_inv=tf.eye(X.shape[1])+tf.matmul(Y,Y,transpose_a=True)
        Y=2*tf.matmul(Y,tf.linalg.inv(Y_inv))
    return Y


def prox_f(arg,lam):
    # prox of the iota-function of the feasible set
    out=[]
    ind=0
    for i in range(len(sizes)):
        T=tf.reshape(arg[ind:(ind+sizes[i]*n)],[n,sizes[i]])
        T=proj_orth(T)
        out.append(tf.reshape(T,[-1]))
        ind+=sizes[i]*n
        out.append(arg[ind:(ind+sizes[i])])
        ind+=sizes[i]
    last_T=tf.maximum(tf.minimum(arg[ind:],100.),-100.)
    out.append(last_T)
    return tf.concat(out,0)

batch_size=1500
test_batch_size=1500
steps_per_epch=x_train.shape[0]//batch_size
epch=200
sarah_p=1000
ens_full=100

runs=10
springs=[]
ispalms=[]
palms=[]
ipalms=[]
sgds=[]
samples=np.concatenate([x_train,y_train],1)
samples_test=np.concatenate([x_test,y_test],1)
print(samples.shape)

# initialization
init_vals=[]
for i in range(len(sizes)):
    T=tf.random.normal((n,sizes[i]))
    q,r=tf.linalg.qr(T)
    init_vals.append(tf.reshape(q,[-1]))
    init_vals.append(tf.zeros(sizes[i]))
init_vals.append(0.05*tf.random.normal([10*n]))
init_vals=[tf.concat(init_vals,0).numpy()]

mydir='PNN_results'
if not os.path.isdir(mydir):
    os.mkdir(mydir)

def record_grad(optimizer):
    # computes after each epoch the norm of the Riemannian gradient
    record_ds=tf.data.Dataset.from_tensor_slices(samples).batch(batch_size)
    optimizer.precompile()    
    grad_norms=[]
    for epoch in range(epch):
        optimizer.exec_epoch()
        grad_norm=0.
        grad=0.
        for batch in record_ds:
            grad+=optimizer.model.grad_batch(batch,0)
        ind=0
        params=optimizer.model.X[0]
        for i in range(len(sizes)):
            grad_T=tf.reshape(grad[ind:(ind+sizes[i]*n)],[n,sizes[i]])
            T=tf.reshape(params[ind:(ind+sizes[i]*n)],[n,sizes[i]])
            W_hat=tf.linalg.matmul(grad_T,tf.transpose(T))-0.5*tf.linalg.matmul(T,tf.linalg.matmul(tf.transpose(T),tf.linalg.matmul(grad_T,tf.transpose(T))))
            W=W_hat-tf.transpose(W_hat)
            rie_grad=tf.linalg.matmul(W,T)
            grad_norm+=tf.reduce_sum(rie_grad**2)
            ind+=sizes[i]*n
            grad_norm+=tf.reduce_sum(grad[ind:(ind+sizes[i])]**2)
            ind+=sizes[i]
        grad_norm+=tf.reduce_sum(grad[ind:]**2)
        grad_norm=grad_norm.numpy()
        grad_norms.append(grad_norm)
        print(grad_norm)
    return optimizer.my_times,optimizer.test_vals,optimizer.train_vals,grad_norms


for run in range(0,runs):
    np.random.seed(10+2*run)
    tf.random.set_seed(11+2*run)

    sarah_seq=tf.random.uniform(shape=[epch*steps_per_epch+100],minval=0,maxval=1,dtype=tf.float32)
    # PALM_Model declaration
    model=PALM_Model(init_vals,dtype='float32')
    model.H=H
    model.prox_funs=[prox_f]
    model2=PALM_Model(init_vals,dtype='float32')
    model3=PALM_Model(init_vals,dtype='float32')
    model4=PALM_Model(init_vals,dtype='float32')
    model2.H=H
    model2.prox_funs=[prox_f]
    model3.H=H
    model3.prox_funs=[prox_f]
    model4.H=H
    model4.prox_funs=[prox_f]

    # run algorithms
    print('\n-------------------- RUN '+str(run+1)+' SPRING --------------------\n')
    spring_optimizer=PALM_Optimizer(model,data=samples,batch_size=batch_size,method='SPRING-SARAH',step_size=0.7,steps_per_epoch=steps_per_epch,sarah_seq=sarah_seq,ensure_full=ens_full,backup_dir=None,sarah_p=sarah_p,test_data=samples_test,test_batch_size=test_batch_size)
    spring=record_grad(spring_optimizer)
    spring=spring+(accuracy(model.X,samples_test),)
    with open(mydir+'/spring'+str(run)+'.pickle', 'wb') as f:
        pickle.dump(spring,f)
    springs.append(spring)
    print('\n-------------------- RUN '+str(run+1)+' iSPALM --------------------\n')
    ispalm_optimizer=PALM_Optimizer(model2,data=samples,batch_size=batch_size,method='iSPALM-SARAH',step_size=.7,inertial_step_size=.99,steps_per_epoch=steps_per_epch,sarah_seq=sarah_seq,ensure_full=ens_full,backup_dir=None,sarah_p=sarah_p,test_data=samples_test,test_batch_size=test_batch_size)
    ispalm=record_grad(ispalm_optimizer)
    ispalm=ispalm+(accuracy(model2.X,samples_test),)
    with open(mydir+'/ispalm'+str(run)+'.pickle', 'wb') as f:
        pickle.dump(ispalm,f)
    ispalms.append(ispalm)
    
    if run==0:
        
        print('\n-------------------- RUN '+str(run+1)+' PALM --------------------\n')
        palm_optimizer=PALM_Optimizer(model3,data=samples,batch_size=batch_size,method='PALM',step_size=1.,backup_dir=None,test_data=samples_test)
        palm=record_grad(palm_optimizer)
        palm=palm+(accuracy(model3.X,samples_test),)
        with open(mydir+'/palm'+str(run)+'.pickle', 'wb') as f:
            pickle.dump(palm,f)
        palms.append(palm)
        
        print('\n-------------------- RUN '+str(run+1)+' iPALM --------------------\n')
        ipalm_optimizer=PALM_Optimizer(model4,data=samples,batch_size=batch_size,method='iPALM',step_size=.9,backup_dir=None,test_data=samples_test)
        ipalm=record_grad(ipalm_optimizer)
        ipalm=ipalm+(accuracy(model4.X,samples_test),)
        with open(mydir+'/ipalm'+str(run)+'.pickle', 'wb') as f:
            pickle.dump(ipalm,f)
        ipalms.append(ipalm)



av_acc_palm=np.mean([p[4] for p in palms])
av_acc_ipalm=np.mean([p[4] for p in ipalms])
av_acc_spring=np.mean([p[4] for p in springs])
av_acc_ispalm=np.mean([p[4] for p in ispalms])

print('Average accuracy PALM:   '+str(av_acc_palm))
print('Average accuracy iPALM:  '+str(av_acc_ipalm))
print('Average accuracy SPRING: '+str(av_acc_spring))
print('Average accuracy iSPALM: '+str(av_acc_ispalm))

av_grad_palm=np.mean(np.stack([p[3] for p in palms]),axis=0)
av_grad_ipalm=np.mean(np.stack([p[3] for p in ipalms]),axis=0)
av_grad_spring=np.mean(np.stack([p[3] for p in springs]),axis=0)
av_grad_ispalm=np.mean(np.stack([p[3] for p in ispalms]),axis=0)

fig=plt.figure()
plt.plot(av_grad_palm,'-',c='red')
plt.plot(av_grad_ipalm,'--',c='green')
plt.plot(av_grad_spring,'-.',c='black')
plt.plot(av_grad_ispalm,':',c='blue')
plt.yscale("log")
plt.legend(['PALM','iPALM','SPRING-SARAH','iSPALM-SARAH'])
fig.savefig(mydir+'/PNNs_grads.png',dpi=1200)
plt.close(fig)

av_steps_palm=np.mean(np.stack([p[2] for p in palms]),axis=0)/np.prod(y_train.shape)
av_steps_ipalm=np.mean(np.stack([p[2] for p in ipalms]),axis=0)/np.prod(y_train.shape)
av_steps_spring=np.mean(np.stack([p[2] for p in springs]),axis=0)/np.prod(y_train.shape)
av_steps_ispalm=np.mean(np.stack([p[2] for p in ispalms]),axis=0)/np.prod(y_train.shape)
if runs>1:
    std_steps_spring=np.sqrt(np.mean((np.stack([p[2]/np.prod(y_train.shape) for p in springs])-av_steps_spring)**2,axis=0))
    std_steps_ispalm=np.sqrt(np.mean((np.stack([p[2]/np.prod(y_train.shape) for p in ispalms])-av_steps_ispalm)**2,axis=0))

av_steps_palm_test=np.mean(np.stack([p[1] for p in palms]),axis=0)/np.prod(y_test.shape)
av_steps_ipalm_test=np.mean(np.stack([p[1] for p in ipalms]),axis=0)/np.prod(y_test.shape)
av_steps_spring_test=np.mean(np.stack([p[1] for p in springs]),axis=0)/np.prod(y_test.shape)
av_steps_ispalm_test=np.mean(np.stack([p[1] for p in ispalms]),axis=0)/np.prod(y_test.shape)

# Plot results
fig=plt.figure()
plt.plot(av_steps_palm,'-',c='red')
plt.plot(av_steps_ipalm,'--',c='green')
plt.plot(av_steps_spring,'-.',c='black')
plt.plot(av_steps_ispalm,':',c='blue')
plt.yscale("log")
plt.legend(['PALM','iPALM','SPRING-SARAH','iSPALM-SARAH'])
fig.savefig(mydir+'/PNNs_train.png',dpi=1200)
plt.close(fig)

if runs>1:
    fig=plt.figure()
    plt.plot(std_steps_spring,'-.',c='black')
    plt.plot(std_steps_ispalm,':',c='blue')
    plt.legend(['SPRING-SARAH','iSPALM-SARAH'])
    fig.savefig(mydir+'/PNNs_train_std.png',dpi=1200)
    plt.close(fig)

fig=plt.figure()
plt.plot(av_steps_palm_test,'-',c='red')
plt.plot(av_steps_ipalm_test,'--',c='green')
plt.plot(av_steps_spring_test,'-.',c='black')
plt.plot(av_steps_ispalm_test,':',c='blue')
plt.yscale("log")
plt.legend(['PALM','iPALM','SPRING-SARAH','iSPALM-SARAH'])
fig.savefig(mydir+'/PNNs_test.png',dpi=1200)
plt.close(fig)

all_end_times=[p[0][-1] for p in palms+ipalms+springs+ispalms]
my_end_time=np.min(all_end_times)

def average_times(data,end_time):
    times=np.sort(np.concatenate([p[0] for p in data],0))
    times=times[times<=end_time]
    inds=[0]*len(data)
    vals=[]
    vals_train=[]
    for t in times:
        obj=0.
        obj_train=0.
        for i in range(len(data)):
            p=data[i]
            times_p=p[0]
            while times_p[inds[i]+1]<t:
                inds[i]+=1
            t_low=times_p[inds[i]]
            t_high=times_p[inds[i]+1]
            coord=(t-t_low)/(t_high-t_low)
            obj+=((1-coord)*p[1][inds[i]]+coord*p[1][inds[i]+1])/np.prod(x_test.shape)
            obj_train+=((1-coord)*p[2][inds[i]]+coord*p[2][inds[i]+1])/np.prod(x_train.shape)
        vals.append(obj/len(data))
        vals_train.append(obj_train/len(data))
    return times,vals,vals_train

t_palm,v_test_palm,v_palm=average_times(palms,my_end_time)
t_ipalm,v_test_ipalm,v_ipalm=average_times(ipalms,my_end_time)
t_spring,v_test_spring,v_spring=average_times(springs,my_end_time)
t_ispalm,v_test_ispalm,v_ispalm=average_times(ispalms,my_end_time)

fig=plt.figure()
plt.plot(t_palm,v_palm,'-',c='red')
plt.plot(t_ipalm,v_ipalm,'--',c='green')
plt.plot(t_spring,v_spring,'-.',c='black')
plt.plot(t_ispalm,v_ispalm,':',c='blue')
plt.yscale("log")
plt.legend(['PALM','iPALM','SPRING-SARAH','iSPALM-SARAH'])
fig.savefig(mydir+'/PNNs_train_times.png',dpi=1200)
plt.close(fig)

fig=plt.figure()
plt.plot(t_palm,v_test_palm,'-',c='red')
plt.plot(t_ipalm,v_test_ipalm,'--',c='green')
plt.plot(t_spring,v_test_spring,'-.',c='black')
plt.plot(t_ispalm,v_test_ispalm,':',c='blue')
plt.yscale("log")
plt.legend(['PALM','iPALM','SPRING-SARAH','iSPALM-SARAH'])
fig.savefig(mydir+'/PNNs_test_times.png',dpi=1200)
plt.close(fig)

