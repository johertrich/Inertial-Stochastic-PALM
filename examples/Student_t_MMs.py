# This code belongs to the paper
#
# J. Hertrich and G. Steidl. 
# Inertial Stochastic PALM and Applications in Machine Learning. 
# Sampling Theory, Signal Processing, and Data Analysis, 2022.
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
import sklearn
import sklearn.cluster
import pickle
import os

np.random.seed(11)
tf.random.set_seed(12)

# Basic functions for sampling from mixture models and initialization
def generate_MM_parameters_random(my_K):
    sigmas=[]
    nus=[]
    mus=[]
    alphas=np.random.normal(size=my_K)
    alphas=alphas*alphas+1
    alphas=alphas/np.sum(alphas)
    for k in range(my_K):
        sigma=np.random.normal(scale=1,size=(d,d))
        sigma=sigma.dot(sigma.transpose())+np.eye(d)
        sigmas.append(sigma)
        mus.append(np.random.normal(scale=2,size=d))
        nu=np.random.normal(scale=10)
        nus.append(np.minimum(nu*nu+1,100))
    return alphas,mus,nus,sigmas

def generate_MM_parameters_small():
    alphas=np.array([.5,1./3.,1./6.])
    mus=[np.array([1.,0.]),np.array([0.,1.]),np.array([1.,1.])]
    nus=[1,2,5]
    sigmas=[np.array([[1.,0.],[0.,2.]]),np.array([[3.,0.],[0.,2.]]),np.array([[4.,1.],[1.,2.]])]
    return alphas,mus,nus,sigmas

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
        print(ni)
        alphas[k]=ni*1./n
        samps_k=samples[labels==k,:]
        mu_gauss=1./ni*np.sum(samps_k,axis=0)
        cent_samps=samps_k-np.matlib.repmat(mu_gauss,ni,1)
        Sigma_gauss=1./ni*cent_samps.transpose().dot(cent_samps)
        nu,mu,Sigma=opt_em_var_many_data(samps_k,3,mu_gauss,Sigma_gauss,regularize=1e-5,epch=100,mute=True,batch_size=100000)
        nus.append(nu)
        mus.append(mu)
        Sigmas.append(Sigma)
    return alphas,nus,mus,Sigmas

def opt_em_var_many_data(samps,nu_init,mu_init,Sigma_init,regularize=1e-5,epch=100,mute=True,batch_size=100000):
    # Variant of the EM algorithm for estimating the parameters of a single Student-t distribution.
    # For the derivation and convergence proof of this algorithm see
    #
    # M. Hasannasab, J. Hertrich, F. Laus and G. Steidl (2020).
    # Alternatives to the EM Algorithm for ML-Estimation of Location, Scatter Matrix and Degree of Freedom of the Student-t Distribution.
    # Numerical Algorithms.
    #
    n=samps.shape[0]
    d=samps.shape[1]
    tic=time.time()
    @tf.function
    def EM_var_step(inputs,nu,mu,Sigma):
        n_inp=inputs.shape[0]
        Sigma_inv=tf.linalg.inv(Sigma)
        cent_inp=inputs-tf.tile(tf.expand_dims(mu,0),(n_inp,1))
        deltas=tf.reduce_sum(tf.matmul(cent_inp,Sigma_inv)*cent_inp,1)
        factor=tf.math.lgamma(0.5*(nu+d))-tf.math.lgamma(0.5*nu)-0.5*d*(tf.math.log(tf.constant(math.pi,dtype=tf.float64))+tf.math.log(nu))-0.5*tf.linalg.logdet(Sigma)-0.5*(nu+d)*tf.math.log1p(deltas/nu)
        log_fun_vals=factor
        gammas=(nu+d)/(nu+deltas)

        mu_new_oben=tf.reduce_sum(tf.tile(gammas[:,tf.newaxis],(1,d))*inputs,axis=0)
        Sigma_mu_new_unten=tf.reduce_sum(gammas)
        Sigma_new_oben=tf.matmul(tf.tile(gammas[:,tf.newaxis],(1,d))*cent_inp,cent_inp,transpose_a=True)
        return mu_new_oben,Sigma_new_oben,Sigma_mu_new_unten
    @tf.function
    def nu_constant(inputs,nu,mu,Sigma):
        n_inp=inputs.shape[0]        
        Sigma_inv=tf.linalg.inv(Sigma)
        cent_inp=inputs-tf.tile(tf.expand_dims(mu,0),(n_inp,1))
        deltas=tf.reduce_sum(tf.matmul(cent_inp,Sigma_inv)*cent_inp,1)
        gammas=(nu+d)/(nu+deltas)
        return tf.reduce_sum(gammas-tf.math.log(gammas))/n
        

    def nu_EM_var_step(nu_in,const,tol=1e-5,max_steps=1000):
        const=const-1-tf.math.digamma((nu_in+d)/2)+tf.math.log((nu_in+d)/2)
        f=lambda x: tf.math.digamma(x/2)-tf.math.log(x/2)+const
        der_f=lambda x:.5*tf.math.polygamma(1,x/2)-1/x
        zero=nu_in
        stps=0
        f_zero=f(zero)
        eps=tf.abs(f_zero)
        while stps<max_steps and eps>=tol:
            newzero=zero-f_zero/der_f(zero)
            zero=tf.math.maximum(newzero,tf.constant(regularize,dtype=tf.float64))
            f_zero=f(zero)
            eps=tf.abs(f_zero)
            stps+=1
        if eps>=tol:
            print('newton not converged'+str(zero.numpy())+' '+str(f_zero.numpy())+' const ' + str(const.numpy()))
        return zero
    
    nu=tf.constant(nu_init,dtype=tf.float64)
    mu=tf.constant(mu_init,dtype=tf.float64)
    Sigma=tf.constant(Sigma_init,dtype=tf.float64)
    for epoch in range(epch):
        ds=tf.data.Dataset.from_tensor_slices(samps).batch(batch_size)
        mu_new_oben=0.
        Sigma_new_oben=0.
        Sigma_mu_new_unten=0.
        for smps in ds:
            out=EM_var_step(smps,nu,mu,Sigma)
            mu_new_oben+=out[0]
            Sigma_new_oben+=out[1]
            Sigma_mu_new_unten+=out[2]
        mu_new=mu_new_oben/Sigma_mu_new_unten
        Sigma_new=Sigma_new_oben/Sigma_mu_new_unten
        const=0.
        for smps in ds:
            const+=nu_constant(smps,nu,mu_new,Sigma_new)      
        nu_new=nu_EM_var_step(nu,const)
        nu_new=tf.math.minimum(nu_new,tf.constant(200.,dtype=tf.float64))
        eps=tf.reduce_sum((tf.identity(nu)-tf.identity(nu_new))**2)
        eps+=tf.reduce_sum((tf.identity(mu)-tf.identity(mu_new))**2)
        eps+=tf.reduce_sum((tf.identity(Sigma)-tf.identity(Sigma_new))**2)
        if not mute:
            print('Value in step '+str(epoch) +': '+str(objective.numpy())+' Time: '+str(time.time()-tic)+' Change: '+str(eps.numpy()))
        mu=mu_new
        nu=nu_new
        Sigma=Sigma_new
        if eps<1e-5:
            return nu,mu,Sigma
    return nu,mu,Sigma

d=10
K=30
#d=2
#K=3
reg_epsilon=1e-1

# required functions for PALM models
n_data=200000

@tf.function
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
    log_fun_vals=[]
    for k in range(0,K):
        nu=nus_tilde[k]*nus_tilde[k]+reg_epsilon
        mu=mus[k]
        Sigma=tf.matmul(tf.transpose(Sigmas_tilde[k]),Sigmas_tilde[k])+reg_epsilon*tf.cast(tf.eye(d), dtype=tf.float64)
        sig_prob=False        
        try:
            Sigma_inv=tf.linalg.inv(Sigma)
        except:
            print(Sigmas_tilde)
            print(Sigma)
            #print(tf.math.logdet(Sigma))
            sig_prob=True
        if sig_prob:
            raise ValueError('Sigma contains NaNs!')
        cent_inp=inputs-tf.tile(tf.expand_dims(mu,0),(n,1))
        deltas=tf.reduce_sum(tf.matmul(cent_inp,Sigma_inv)*cent_inp,1)
        factor=tf.math.lgamma(0.5*(nu+d))-tf.math.lgamma(0.5*nu)-0.5*d*(tf.math.log(tf.constant(math.pi,dtype=tf.float64))+tf.math.log(nu))-0.5*tf.linalg.logdet(Sigma)-0.5*(nu+d)*tf.math.log1p(deltas/nu)
        log_fun_vals.append(factor)
    log_fun_vals=tf.stack(log_fun_vals)
    log_alphas=tf.tile(tf.expand_dims(tf.math.log(alphas),-1),(1,n))
    log_fun_vals2=log_alphas+log_fun_vals
    const=tf.reduce_max(log_fun_vals2,0)
    log_fun_vals2-=tf.tile(tf.expand_dims(const,0),(K,1))
    log_fun_vals2=tf.math.log(tf.reduce_sum(tf.exp(log_fun_vals2),0))+const
    y=-tf.reduce_sum(log_fun_vals2)/n_data
    return y



n=n_data

batch_size=20000
steps_per_epch=10
epch=50
sarah_p=1000
ens_full=steps_per_epch
print(steps_per_epch)

runs=100
springs=[]
ispalms=[]
palms=[]
ipalms=[]

# build the model
def myprox_alpha(arg,lam,min_val=0.,max_val=10.):
    return tf.minimum(tf.maximum(arg,min_val),max_val)

def student_t_MM_palm_model(init_vals):
    mod=PALM_Model(init_vals,dtype='float64')
    mod.H=H
    mod.prox_funs[0]=myprox_alpha
    return mod

# record gradients
def record_grad(optimizer):
    record_ds=tf.data.Dataset.from_tensor_slices(samples).batch(batch_size)
    optimizer.precompile()    
    grad_norms=[]
    for epoch in range(epch):
        optimizer.exec_epoch()
        grad_norm=0.
        for i in range(0,optimizer.model.num_blocks):
            grad=0
            for batch in record_ds:
                grad+=optimizer.model.grad_batch(batch,i).numpy()
            if i==0:
                grad[model.X[0].numpy()<1e-3]=np.minimum(grad[model.X[0].numpy()<1e-3],0.)
                grad[model.X[0].numpy()>10-1e-3]=np.maximum(grad[model.X[0].numpy()>10-1e-3],0.)
            grad_norm+=np.sum(grad**2)
        grad_norms.append(grad_norm)
        print(grad_norm)
    return optimizer.my_times,optimizer.test_vals,grad_norms

# initialization
alphas,mus,nus,sigmas=generate_MM_parameters_random(K)

samples,true_labels=mixture_sample(alphas,nus,mus,sigmas,n)

alphas_init,nus_init,mus_init,Sigmas_init=init_labels(samples,np.random.randint(0,K,samples.shape[0]),K)
Sigmas_init=tf.maximum(tf.minimum(Sigmas_init,100.),-100.)
init_Sigma=[]
real_Sigma=[]
for k in range(K):
    b,a=np.linalg.eig(Sigmas_init[k])
    b=np.maximum(0,b-reg_epsilon)
    sqrt_sigma_init=a.dot(np.diag(np.sqrt(b)).dot(a.transpose()))
    init_Sigma.append(sqrt_sigma_init)
    b,a=np.linalg.eig(sigmas[k])
    b=np.maximum(0,b-reg_epsilon)
    sqrt_sigma_init=a.dot(np.diag(np.sqrt(b)).dot(a.transpose()))
    real_Sigma.append(sqrt_sigma_init)
init_vals=[np.log(alphas_init),np.sqrt(np.array(nus_init))-reg_epsilon,np.array(mus_init),np.array(init_Sigma)]
real_vals=[np.log(alphas),np.sqrt(np.array(nus))-reg_epsilon,np.array(mus),np.array(real_Sigma)]
for val in init_vals:
    val=val.astype(np.float64)
samples=samples.astype(np.float64)
print(H(real_vals,samples))

for run in range(runs):
    sarah_seq=tf.random.uniform(shape=[epch*steps_per_epch*4+100],minval=0,maxval=1,dtype=tf.float32)

    model=student_t_MM_palm_model(init_vals)
    model2=student_t_MM_palm_model(init_vals)
    model3=student_t_MM_palm_model(init_vals)
    model4=student_t_MM_palm_model(init_vals)

    # run algorithms
    print('\n-------------------- RUN '+str(run+1)+' SPRING --------------------\n')
    spring_optimizer=PALM_Optimizer(model,data=samples,batch_size=batch_size,method='SPRING-SARAH',step_size=0.7,steps_per_epoch=steps_per_epch,sarah_seq=sarah_seq,ensure_full=ens_full,backup_dir=None,sarah_p=sarah_p)
    spring=record_grad(spring_optimizer)
    print(tf.exp(model.X[0])/tf.reduce_sum(tf.exp(model.X[0])))
    print('\n-------------------- RUN '+str(run+1)+' iSPALM --------------------\n')
    ispalm_optimizer=PALM_Optimizer(model2,data=samples,batch_size=batch_size,method='iSPALM-SARAH',step_size=0.7,inertial_step_size=.8,steps_per_epoch=steps_per_epch,sarah_seq=sarah_seq,ensure_full=ens_full,backup_dir=None,sarah_p=sarah_p)
    ispalm=record_grad(ispalm_optimizer)
    print(tf.exp(model2.X[0])/tf.reduce_sum(tf.exp(model2.X[0])))
    if run==0:
        print('\n-------------------- RUN '+str(run+1)+' PALM --------------------\n')
        palm_optimizer=PALM_Optimizer(model3,data=samples,batch_size=batch_size,method='PALM',step_size=1,backup_dir=None)
        palm=record_grad(palm_optimizer)
        print(tf.exp(model3.X[0])/tf.reduce_sum(tf.exp(model3.X[0])))
        print('\n-------------------- RUN '+str(run+1)+' iPALM --------------------\n')
        ipalm_optimizer=PALM_Optimizer(model4,data=samples,batch_size=batch_size,method='iPALM',step_size=1.,inertial_step_size=1.,backup_dir=None)
        ipalm=record_grad(ipalm_optimizer)
        print(tf.exp(model4.X[0])/tf.reduce_sum(tf.exp(model4.X[0])))
        palms.append(palm)
        ipalms.append(ipalm)

    springs.append(spring)
    ispalms.append(ispalm)


mydir='Student_t_results'
#save results
if not os.path.isdir(mydir):
    os.mkdir(mydir)
with open(mydir+'/springs.pickle', 'wb') as f:
    pickle.dump(springs,f)
with open(mydir+'/ispalms.pickle', 'wb') as f:
    pickle.dump(ispalms,f)
with open(mydir+'/palms.pickle', 'wb') as f:
    pickle.dump(palms,f)
with open(mydir+'/ipalms.pickle', 'wb') as f:
    pickle.dump(ipalms,f)

av_grads_ispalm=np.mean(np.stack([p[2] for p in ispalms]),axis=0)
av_grads_ipalm=np.mean(np.stack([p[2] for p in ipalms]),axis=0)
av_grads_palm=np.mean(np.stack([p[2] for p in palms]),axis=0)
av_grads_spring=np.mean(np.stack([p[2] for p in springs]),axis=0)

# Plot results
fig=plt.figure()
plt.plot(av_grads_palm,'-',c='red')
plt.plot(av_grads_ipalm,'--',c='green')
plt.plot(av_grads_spring,'-.',c='black')
plt.plot(av_grads_ispalm,':',c='blue')
plt.yscale("log")
plt.legend(['PALM','iPALM','SPRING-SARAH','iSPALM-SARAH'])
fig.savefig('Student_t_results/Student_t_MMs_grads.png',dpi=1200)
plt.close(fig)
av_steps_palm=np.mean(np.stack([p[1] for p in palms]),axis=0)
av_steps_ipalm=np.mean(np.stack([p[1] for p in ipalms]),axis=0)
av_steps_spring=np.mean(np.stack([p[1] for p in springs]),axis=0)
av_steps_ispalm=np.mean(np.stack([p[1] for p in ispalms]),axis=0)

std_steps_spring=np.sqrt(np.mean((np.stack([p[1] for p in springs])-av_steps_spring)**2,axis=0))
std_steps_ispalm=np.sqrt(np.mean((np.stack([p[1] for p in ispalms])-av_steps_ispalm)**2,axis=0))

fig=plt.figure()
plt.plot(av_steps_palm,'-',c='red')
plt.plot(av_steps_ipalm,'--',c='green')
plt.plot(av_steps_spring,'-.',c='black')
plt.plot(av_steps_ispalm,':',c='blue')
plt.legend(['PALM','iPALM','SPRING-SARAH','iSPALM-SARAH'])
fig.savefig(mydir+'/Student_t_MMs.png',dpi=1200)
plt.close(fig)

fig=plt.figure()
plt.plot(std_steps_spring,'-.',c='black')
plt.plot(std_steps_ispalm,':',c='blue')
plt.legend(['SPRING-SARAH','iSPALM-SARAH'])
fig.savefig(mydir+'/Student_t_MMs_std.png',dpi=1200)
plt.close(fig)

all_end_times=[p[0][-1] for p in palms+ipalms+springs+ispalms]
my_end_time=np.min(all_end_times)

def average_times(data,end_time):
    times=np.sort(np.concatenate([p[0] for p in data],0))
    times=times[times<=end_time]
    inds=[0]*len(data)
    vals=[]
    for t in times:
        obj=0.
        for i in range(len(data)):
            p=data[i]
            times_p=p[0]
            while times_p[inds[i]+1]<t:
                inds[i]+=1
            t_low=times_p[inds[i]]
            t_high=times_p[inds[i]+1]
            coord=(t-t_low)/(t_high-t_low)
            obj+=(1-coord)*p[1][inds[i]]+coord*p[1][inds[i]+1]
        vals.append(obj/len(data))
    return times,vals

t_palm,v_palm=average_times(palms,my_end_time)
t_ipalm,v_ipalm=average_times(ipalms,my_end_time)
t_spring,v_spring=average_times(springs,my_end_time)
t_ispalm,v_ispalm=average_times(ispalms,my_end_time)

fig=plt.figure()
plt.plot(t_palm,v_palm,'-',c='red')
plt.plot(t_ipalm,v_ipalm,'--',c='green')
plt.plot(t_spring,v_spring,'-.',c='black')
plt.plot(t_ispalm,v_ispalm,':',c='blue')
plt.legend(['PALM','iPALM','SPRING-SARAH','iSPALM-SARAH'])
fig.savefig(mydir+'/Student_t_MMs_times.png',dpi=1200)
plt.close(fig)





