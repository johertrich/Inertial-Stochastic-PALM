# This code belongs to the paper
#
# J. Hertrich and G. Steidl. 
# Inertial Stochastic PALM and its Application for Learning Student-t Mixture Models. 
# ArXiv preprint arXiv:2005.02204, 2020.
#
# Please cite the paper, if you use the code.
#
# This function implements the sparse PCA using an implementation PALM, iPALM, SPRING-SARAH and iSPRING-SARAH 
# from scratch for comparing the perfomance of these algorithms.
# The general framework of PALM, iPALM, SPRING-SARAH and iSPRING-SARAH is implemented in palm_algs.py
# The implementation of the sparse PCA using this framework is given in sparse_PCA_palm_algs.py

import math
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import numpy.matlib
import numpy.random
import matplotlib.pyplot as plt
import time

class sparse_PCA(Model):
    def __init__(self,A=None,X_init=None,Y_init=None,r=None,d=None,n=None,lambda1=None,lambda2=None):
        super(sparse_PCA,self).__init__()
        if A is None:
            self.A=50*tf.random.uniform(shape=[n,d])
        else:
            self.A=A
            n=A.shape[0]
            d=A.shape[1]
        if X_init is None:
            X_init=20*tf.random.uniform(shape=[n,r]).numpy()
        else:
            r=X_init.shape[1]
        if Y_init is None:
            Y_init=20*tf.random.uniform(shape=[r,d]).numpy()
        if lambda1 is None:
            self.lambda1=tf.constant(0.)
        else:
            self.lambda1=tf.constant(lambda1,dtype=tf.float32)
        if lambda2 is None:
            self.lambda2=tf.constant(0.)
        else:
            self.lambda2=tf.constant(lambda2,dtype=tf.float32)
        self.factor=tf.constant(0.25)
        self.inertial_factor=tf.constant(0.5)
        init_X=tf.constant_initializer(X_init)
        init_Y=tf.constant_initializer(Y_init)
        self.X=self.add_weight("X",initializer=init_X,shape=[n,r],trainable=True)
        self.Y=self.add_weight("Y",initializer=init_Y,shape=[r,d],trainable=True)
        

    def call(self,A=None,X=None,Y=None):
        if A is None:
            A=self.A
        if X is None:
            X=self.X
        if Y is None:
            Y=self.Y
        inner=A-tf.matmul(X,Y)
        out=tf.reduce_sum(inner*inner)
        return out

    def objective(self):
        return self.call+tf.reduce_sum(tf.math.abs(self.X))+tf.reduce_sum(tf.math.abs(self.Y))

def opt_sparse_PCA(model,epch=10,batch_size=1024,method='SPRING',steps_per_epoch=np.inf,spring_p_inv=None,spring_seq=None):
    n=model.A.shape[0]
    if spring_p_inv is None:
        spring_p_inv=1000.0/n
    model.X_old=model.X
    model.Y_old=model.Y
    train_loss=tf.keras.metrics.Mean(name='train_loss')
    train_loss_step=tf.keras.metrics.Mean(name='train_loss')
    val=model.call().numpy()
    test_vals=[val]
    times=[0.]
    my_time=0.
    print('Initial obj: '+str(test_vals[0]))

    def train_step_iSPRING_SARAH(batch_ind,steps,inertial=False,full=False):
        bs=len(batch_ind)
        if inertial:
            extr=model.inertial_factor*(steps-1.)/(steps+2.)
        else:
            extr=0
        if full:
            batch_ind=np.array(range(n))
            bs=n
        #batch_ind=np.sort(batch_ind)
        #print(batch_ind)
        A_batch=tf.gather(model.A,batch_ind)
        X_save=tf.identity(model.X)
        model.X.assign(model.X+extr*(model.X-model.X_old))
        with tf.GradientTape(persistent=True) as tape:
            X_batch=tf.gather(model.X,batch_ind)
            val=model.call(A=A_batch,X=X_batch)
            gradient=tf.identity(tape.gradient(val,model.X))
            grad_sum=tf.reduce_sum(gradient*gradient)
        hess=0.5*tf.identity(tape.gradient(grad_sum,model.X))
        hess=hess/tf.sqrt(grad_sum)
        Lip=tf.sqrt(tf.reduce_sum(hess*hess))
        model.tau_X=Lip*n/bs
        if gradient is None:
            return
        if not full:
            with tf.GradientTape() as tape:
                tape.watch(model.X_old_arg)
                X_old_arg=tf.gather(model.X_old_arg,batch_ind)
                val=model.call(A=A_batch,X=X_old_arg,Y=model.Y_old)
            gradient2=tape.gradient(val,model.X_old_arg)
            gradient=n/bs*(gradient-gradient2)+model.gradient_X
        model.gradient_X=tf.identity(gradient)
        model.X_old_arg=tf.identity(model.X)
        train_loss(val)
        train_loss_step(val)
        model.X.assign_sub(gradient/model.tau_X*model.factor)
        model.X.assign(tf.math.sign(model.X)*tf.math.maximum(0.,tf.math.abs(model.X)-model.lambda1/model.tau_X))
        model.X_old=X_save
        
        Y_save=tf.identity(model.Y)
        model.Y.assign(model.Y+extr*(model.Y-model.Y_old))
        X_batch=tf.gather(model.X,batch_ind)
        with tf.GradientTape(persistent=True) as tape:
            val=model.call(A=A_batch,X=X_batch,Y=model.Y)
            gradient=tf.identity(tape.gradient(val,model.Y))
            grad_sum=tf.reduce_sum(gradient*gradient)
        hess=0.5*tf.identity(tape.gradient(grad_sum,model.Y))
        hess=hess/tf.sqrt(grad_sum)
        Lip=tf.sqrt(tf.reduce_sum(hess*hess))
        model.tau_Y=Lip*n/bs
        if gradient is None:
            return
        if not full:
            X_old_batch=tf.gather(model.X_old,batch_ind)
            with tf.GradientTape() as tape:
                tape.watch(model.Y_old_arg)
                val=model.call(A=A_batch,X=X_old_batch,Y=model.Y_old_arg)
            gradient2=tape.gradient(val,model.Y_old_arg)
            gradient=n/bs*(gradient-gradient2)+model.gradient_Y
        model.gradient_Y=tf.identity(gradient)
        model.Y_old_arg=tf.identity(model.Y)
        model.Y.assign_sub(gradient/model.tau_Y*model.factor)
        model.Y.assign(tf.math.sign(model.Y)*tf.math.maximum(0.,tf.math.abs(model.Y)-model.lambda2/model.tau_Y))
        model.Y_old=Y_save
    
    step=0
    ds=tf.data.Dataset.from_tensor_slices(np.array(range(n))).shuffle(n).batch(batch_size)
    for epoch in range(epch):
        print('Epoch '+str(epoch+1)+':')
        count=0
        for batch in ds:
            step+=1
            count+=1
            if count>=steps_per_epoch:
                break
            if method=='SPRING':
                if spring_seq is None:                
                    rand_num=tf.random.uniform(shape=[1])
                else:
                    rand_num=spring_seq[step]
                full=False
                if rand_num<spring_p_inv:
                    full=True
                if step==1 or (step-1)%20==0:
                    full=True
                tic=time.time()
                train_step_iSPRING_SARAH(batch.numpy(),step,inertial=False,full=full)
                toc=time.time()-tic
                if full:
                    print('.',end='')
            elif method=='iSPRING':
                if spring_seq is None:                
                    rand_num=tf.random.uniform(shape=[1])
                else:
                    rand_num=spring_seq[step]
                full=False
                if rand_num<spring_p_inv:
                    full=True
                if step==1 or (step-1)%20==0:
                    full=True
                tic=time.time()
                train_step_iSPRING_SARAH(batch.numpy(),step,inertial=True,full=full)
                toc=time.time()-tic
                if full:
                    print('.',end='')
            elif method=='PALM':
                tic=time.time()
                train_step_iSPRING_SARAH(batch.numpy(),step,inertial=False,full=True)
                toc=time.time()-tic
            elif method=='iPALM':
                tic=time.time()
                train_step_iSPRING_SARAH(batch.numpy(),step,inertial=True,full=True)
                toc=time.time()-tic
            my_time+=toc
        val=model.call().numpy()
        test_vals.append(val)
        times.append(my_time)
        diff=np.nan
        if len(test_vals)>1:
            diff=test_vals[-1]-test_vals[-2]
        print('Train value: '+str(train_loss.result().numpy())+' Test value: '+str(val)+' Difference: '+str(diff))
    return test_vals,times

if __name__=='__main__':
    # initialization and model parameters
    n=1000000
    d=20
    r=5
    A=50*tf.random.uniform(shape=[n,d])
    X_init=20*tf.random.uniform(shape=[n,r]).numpy()
    Y_init=20*tf.random.uniform(shape=[r,d]).numpy()
    lambda1=0.1
    lambda2=0.1
    steps_per_epoch=10
    #warm start
    model1=sparse_PCA(A=A,X_init=X_init,Y_init=Y_init,lambda1=lambda1,lambda2=lambda2)
    model1.factor=tf.constant(1.)
    opt_sparse_PCA(model1,2,batch_size=n,method='PALM')
    X_init=model1.X.numpy()
    Y_init=model1.Y.numpy()
    #initialization
    tv1=opt_sparse_PCA(model1,50,batch_size=n,method='PALM')
    model2=sparse_PCA(A=A,X_init=X_init,Y_init=Y_init,lambda1=lambda1,lambda2=lambda2)
    model2.factor=tf.constant(1.)
    model2.inertial_factor=tf.constant(1.)
    tv2=opt_sparse_PCA(model2,50,batch_size=n,method='iPALM')
    model3=sparse_PCA(A=A,X_init=X_init,Y_init=Y_init,lambda1=lambda1,lambda2=lambda2)
    model3.factor=tf.constant(0.4)
    tv3=opt_sparse_PCA(model3,50,batch_size=10000,method='SPRING',steps_per_epoch=steps_per_epoch)
    model4=sparse_PCA(A=A,X_init=X_init,Y_init=Y_init,lambda1=lambda1,lambda2=lambda2)
    model4.factor=tf.constant(0.25)
    tv4=opt_sparse_PCA(model4,50,batch_size=10000,method='iSPRING',steps_per_epoch=steps_per_epoch)
    #generate plots
    fig=plt.figure()
    plt.plot(tv1[1],tv1[0],'-',c='red')
    plt.plot(tv2[1],tv2[0],'--',c='green')
    plt.plot(tv3[1][:20],tv3[0][:20],'-.',c='black')
    plt.plot(tv4[1][:20],tv4[0][:20],':',c='blue')
    plt.legend(['PALM','iPALM','SPRING-SARAH','iSPRING-SARAH'])
    fig.savefig('sparse_PCA_by_time_compare.png',dpi=1200)
    plt.close(fig)
    fig=plt.figure()
    plt.plot(tv1[0],'-',c='red')
    plt.plot(tv2[0],'--',c='green')
    plt.plot(tv3[0],'-.',c='black')
    plt.plot(tv4[0],':',c='blue')
    plt.legend(['PALM','iPALM','SPRING-SARAH','iSPRING-SARAH'])
    fig.savefig('sparse_PCA_by_epochs_compare.png',dpi=1200)
    plt.close(fig) 
