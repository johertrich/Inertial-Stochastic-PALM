# This code belongs to the paper
#
# J. Hertrich and G. Steidl. 
# Inertial Stochastic PALM and its Application for Learning Student-t Mixture Models. 
# ArXiv preprint arXiv:2005.02204, 2020.
#
# Please cite the paper, if you use the code.

from palm_algs import *
import numpy.random
import numpy.matlib
import math
import matplotlib.pyplot as plt

# Basic functions for sampling from mixture models and initialization

def mixture_sample(alphas,nus,mus,sigmas,n):
    K=alphas.shape[0]
    d=mus[0].shape[0]
    X=np.zeros((d,n))
    cum_alpha=np.cumsum(alphas)
    zv=np.random.uniform(size=(n))
    classes=np.zeros(n)
    for i in range(0,K-1):
        classes=classes+(zv<cum_alpha[i])
    for i in range(0,K):
        mu=mus[i]
        nu=nus[i]
        sigma=sigmas[i]
        b,a=np.linalg.eig(sigma)
        sqrt_sigma=a.dot(np.diag(np.sqrt(b)).dot(a.transpose()))
        ni=np.sum(classes==i)
        gamma_rands=np.sqrt(np.random.gamma(shape=0.5*nu,scale=2./nu,size=ni))
        X[:,classes==i]=np.matlib.repmat(mu,ni,1).transpose()+(sqrt_sigma.dot(np.random.normal(size=(d,ni))))/np.matlib.repmat(gamma_rands,d,1)
    return X.transpose(),classes

def init_labels(samples,labels,K):
    d=samples.shape[1]
    n=samples.shape[0]
    alphas=np.zeros(K)
    nus=[]
    mus=[]
    Sigmas=[]
    for k in range(K):
        ni=np.sum(labels==k)
        alphas[k]=ni*1./n
        nus.append(3)
        samps_k=samples[labels==k,:]
        mus.append(1./ni*np.sum(samps_k,axis=0))
        cent_samps=samps_k-np.matlib.repmat(mus[k],ni,1)
        Sigmas.append(1./ni*cent_samps.transpose().dot(cent_samps))
    return alphas,nus,mus,Sigmas

d=10
K=30
reg_epsilon=1e-1

# required functions for PALM models

def H(X,batch):
    inputs=batch
    alphas_tilde=X[0]
    nus_tilde=X[1]
    mus=X[2]
    Sigmas_tilde=X[3]
    n=inputs.shape[0]
    if n is None:
        n=1
    alphas=tf.exp(alphas_tilde)
    alphas=alphas/tf.reduce_sum(alphas)
    fun_vals=[]
    for k in range(0,K):
        nu=nus_tilde[k]*nus_tilde[k]+reg_epsilon
        mu=mus[k]
        Sigma=tf.matmul(tf.transpose(Sigmas_tilde[k]),Sigmas_tilde[k])+reg_epsilon*tf.cast(tf.eye(d), dtype=tf.float64)
        sig_prob=False        
        try:
            Sigma_inv=tf.linalg.inv(Sigma)
        except:
            #print(Sigmas_tilde)
            #print(Sigma)
            sig_prob=True
        if sig_prob:
            raise ValueError('Sigma contains NaNs!')
        cent_inp=inputs-tf.tile(tf.expand_dims(mu,0),(n,1))
        deltas=tf.reduce_sum(tf.matmul(cent_inp,Sigma_inv)*cent_inp,1)
        factor=tf.math.lgamma(0.5*(nu+d))-tf.math.lgamma(0.5*nu)-0.5*d*(tf.math.log(tf.constant(math.pi,dtype=tf.float64))+tf.math.log(nu))-0.5*tf.linalg.logdet(Sigma)-0.5*(nu+d)*tf.math.log1p(deltas/nu)
        line=tf.exp(factor)
        fun_vals.append(line)
    fun_vals=tf.stack(fun_vals)
    alphas2=tf.tile(tf.expand_dims(alphas,-1),(1,n))
    fun_vals2=alphas2*fun_vals
    y=-tf.reduce_sum(tf.math.log(tf.reduce_sum(fun_vals2,0)))
    return y


# initialization
sigmas=[]
nus=[]
mus=[]
alphas=np.random.normal(size=K)
alphas=alphas*alphas+1e-3
alphas=alphas/np.sum(alphas)
for k in range(K):
    sigma=np.random.normal(scale=1,size=(d,d))
    sigma=sigma.dot(sigma.transpose())+np.eye(d)
    sigmas.append(sigma)
    mus.append(np.random.normal(scale=2,size=d))
    nu=np.random.normal(scale=10)
    nus.append(np.minimum(nu*nu+0.1,100))
n=100000
samples,true_labels=mixture_sample(alphas,nus,mus,sigmas,n)

alphas_init,nus_init,mus_init,Sigmas_init=init_labels(samples,true_labels,K)
init_Sigma=[]
for k in range(K):
    b,a=np.linalg.eig(Sigmas_init[k])
    b=np.maximum(0,b-reg_epsilon)
    sqrt_sigma_init=a.dot(np.diag(np.sqrt(b)).dot(a.transpose()))
    init_Sigma.append(sqrt_sigma_init)
init_vals=[np.log(alphas_init),np.sqrt(np.array(nus_init))-reg_epsilon,np.array(mus_init),np.array(init_Sigma)]
for val in init_vals:
    val=val.astype(np.float64)
samples=samples.astype(np.float64)

batch_size=10000
steps_per_epch=10
epch=20
sarah_seq=tf.random.uniform(shape=[epch*steps_per_epch*4+100],minval=0,maxval=1,dtype=tf.float32)

# warm start and PALM_Model declaration
model=PALM_Model(init_vals,dtype='float64')
model.H=H
optimize_PALM(model,data=samples,batch_size=batch_size,method='PALM',EPOCHS=2,step_size=1)
init_vals_warm=[model.X[0].numpy(),model.X[1].numpy(),model.X[2].numpy(),model.X[3].numpy()]
model2=PALM_Model(init_vals_warm,dtype='float64')
model3=PALM_Model(init_vals_warm,dtype='float64')
model4=PALM_Model(init_vals_warm,dtype='float64')
model2.H=H
model3.H=H
model4.H=H

# run algorithms
spring=optimize_PALM(model,data=samples,batch_size=batch_size,method='SPRING-SARAH',EPOCHS=epch,step_size=0.7,steps_per_epoch=steps_per_epch,precompile=True,sarah_seq=sarah_seq,ensure_full=True)
ispring=optimize_PALM(model2,data=samples,batch_size=batch_size,method='iSPRING-SARAH',EPOCHS=epch,step_size=0.7,inertial_step_size=0.9,steps_per_epoch=steps_per_epch,precompile=True,sarah_seq=sarah_seq,ensure_full=True)
palm=optimize_PALM(model3,data=samples,batch_size=batch_size,method='PALM',EPOCHS=epch,step_size=1,precompile=True)
ipalm=optimize_PALM(model4,data=samples,batch_size=batch_size,method='iPALM',EPOCHS=epch,step_size=1,precompile=True)

# Plot results

fig=plt.figure()
plt.plot(palm[1],'-',c='red')
plt.plot(ipalm[1],'--',c='green')
plt.plot(spring[1],'-.',c='black')
plt.plot(ispring[1],':',c='blue')
plt.legend(['PALM','iPALM','SPRING-SARAH','iSPRING-SARAH'])
fig.savefig('Student_t_MMs.png',dpi=1200)
plt.close(fig)

end_time=np.min(np.array([palm[0][-1],ipalm[0][-1],spring[0][-1],ispring[0][-1]]))
end_index_palm=1
end_index_ipalm=1
end_index_spring=1
end_index_ispring=1
while end_index_palm<=epch and palm[0][end_index_palm-1]<end_time:
    end_index_palm+=1
while end_index_ipalm<=epch and ipalm[0][end_index_ipalm-1]<end_time:
    end_index_ipalm+=1
while end_index_spring<=epch and spring[0][end_index_spring-1]<end_time:
    end_index_spring+=1
while end_index_ispring<=epch and ispring[0][end_index_ispring-1]<end_time:
    end_index_ispring+=1

fig=plt.figure()
plt.plot(palm[0][:end_index_palm],palm[1][:end_index_palm],'-',c='red')
plt.plot(ipalm[0][:end_index_ipalm],ipalm[1][:end_index_ipalm],'--',c='green')
plt.plot(spring[0][:end_index_spring],spring[1][:end_index_spring],'-.',c='black')
plt.plot(ispring[0][:end_index_ispring],ispring[1][:end_index_ispring],':',c='blue')
plt.legend(['PALM','iPALM','SPRING-SARAH','iSPRING-SARAH'])
fig.savefig('Student_t_MMs_times.png',dpi=1200)
plt.close(fig)





