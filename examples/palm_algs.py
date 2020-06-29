# This code belongs to the paper
#
# J. Hertrich and G. Steidl. 
# Inertial Stochastic PALM and its Application for Learning Student-t Mixture Models. 
# ArXiv preprint arXiv:2005.02204, 2020.
#
# Please cite the paper, if you use the code.

from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import time
import os

class PALM_Model(Model):
    # Models functions of the Form F(x_1,...,x_n)=H(x_1,...,x_n)+\sum_{i=1}^n f_i(x_i),
    # where H is continuously differentiable and f_i is lower semicontinuous.
    # Inputs of the Constructor:
    #       initial_values          - List of numpy arrays for initialize the x_i
    #       dtype                   - model type
    def __init__(self,initial_values,dtype='float32'):
        super(PALM_Model,self).__init__(dtype=dtype)
        self.num_blocks=len(initial_values)
        self.model_type=tf.constant(initial_values[0]).dtype
        self.prox_funs=[]
        self.X=[]
        self.H=lambda X,batch: 0
        self.f=[]
        id_prox=lambda arg,lam:tf.identity(arg)
        for i in range(self.num_blocks):
            self.prox_funs.append(id_prox)
            init=tf.constant_initializer(initial_values[i])
            self.X.append(self.add_weight("X"+str(i),initializer=init,shape=initial_values[i].shape,trainable=True))
            self.f.append(lambda X: 0)

    def call(self,batch,X=None):
        if X is None:
            X=self.X
        #print(batch)
        return self.H(X,batch)
    
    def objective(self,X=None,batch=None):
        if X is None:
            X=self.X
        out=0.
        out+=self(batch)
        for i in range(self.num_blocks):
            out+=self.f[i](X[i])
        return out

        
def optimize_PALM(model,EPOCHS=10,steps_per_epoch=np.inf,data=None,test_data=None,batch_size=1000,method='iSPRING-SARAH',inertial_step_size=None,step_size=None,sarah_seq=None,sarah_p=None,precompile=False,test_batch_size=None,ensure_full=False,estimate_lipschitz=False,backup_dir='backup'):
    # Minimizes the PALM_model using PALM/iPALM/SPRING-SARAH or iSPRING-SARAH.
    # Inputs:
    #       - model                 - PALM_Model for the objective function
    #       - EPOCHS                - int. Number of epochs to optimize
    #                                 Default value: 10
    #       - steps_per_epoch       - int. maximal numbers of PALM/iPALM/SPRING/iSPRING steps in one epoch
    #                                 Default value: Infinity, that is pass the whole data set in each epoch
    #       - data                  - Numpy array of type model.model_type. Information to choose the minibatches. 
    #                                 Required for SPRING and iSPRING. 
    #                                 To run PALM/iPALM on functions, which are not data based, use data=None.
    #                                 For SPRING and iSPRING a value not equal to None is required.
    #                                 Default value: None
    #       - test_data             - Numpy array of type model.model_type. Data points to evaluate the objective   
    #                                 function in the test step after each epoch. 
    #                                 For test_data=None, the function uses data as test_data.
    #                                 Default value: None
    #       - batch_size            - int. If data is None: No effect. Otherwise: batch_size for data driven models.
    #                                 Default value: 1000
    #       - method                - String value, which declares the optimization method. Valid choices are: 'PALM', 
    #                                 'iPALM', 'SPRING-SARAH' and 'iSPRING-SARAH'. Raises an error for other inputs.
    #                                 Default value: 'iSPRING-SARAH'
    #       - inertial_step_size    - float variable. For method=='PALM' or method=='SPRING-SARAH': No effect.      
    #                                 Otherwise: the inertial parameters in iPALM/iSPRING are chosen by 
    #                                 inertial_step_size*(k-1)/(k+2), where k is the current step number.
    #                                 For inertial_step_size=None the method choses 1 for PALM and iPALM, 0.5 for 
    #                                 SPRING and 0.4 for iSPRING.
    #                                 Default value: None
    #       - step_size             - float variable. The step size parameters tau are choosen by step_size*L where L 
    #                                 is the estimated partial Lipschitz constant of H.
    #       - sarah_seq             - This input should be either None or a sequence of uniformly on [0,1] distributed
    #                                 random float32-variables. The entries of sarah_seq determine if the full
    #                                 gradient in the SARAH estimator is evaluated or not.
    #                                 For sarah_seq=None such a sequence is created inside this function.
    #                                 Default value: None
    #       - sarah_p               - float in (1,\infty). Parameter p for the sarah estimator. If sarah_p=None the 
    #                                 method uses p=20
    #                                 Default value: None
    #       - precompile            - Boolean. If precompile=True, then the functions are compiled before the time
    #                                 measurement starts. Otherwise the functions are compiled at the first call.
    #                                 precompiele=True makes only sense if you are interested in the runtime of the
    #                                 algorithms without the compile time of the functions.
    #                                 Default value: False
    #       - test_batch_size       - int. test_batch_size is the batch size used in the test step and in the steps
    #                                 where the full gradient is evaluated. This does not effect the algorithm itself.
    #                                 But it may effect the runtime. For test_batch_size=None it is set to batch_size.
    #                                 If test_batch_size<batch_size and method=SPRING-SARAH or method=iSPRING-SARAH,
    #                                 then also in the steps, where not the full gradient is evaluated only batches
    #                                 of size test_batch_size are passed through the function H.
    #                                 Default value: None
    #       - ensure_full           - Boolean or int. For method=='SPRING-SARAH' or method=='iSPRING-SARAH': If
    #                                 ensure_full is True, we evaluate in the first step of each epoch the full
    #                                 gradient. We observed numerically, that this sometimes increases stability and
    #                                 convergence speed of SPRING and iSPRING. For PALM and iPALM: no effect.
    #                                 If a integer value p is provided, every p-th step is forced to be a full step
    #                                 Default value: False
    #       - estimate_lipschitz    - Boolean. If estimate_lipschitz==True, the Lipschitz constants are estimated based
    #                                 on the first minibatch in all steps, where the full gradient is evaluated.
    #                                 Default: True
    #       - backup_dir            - String or None. If a String is provided, the variables X[i] are saved after
    #                                 every epoch. The weights are not saved if backup_dir is None.
    #                                 Default: 'backup'
    #
    # Outputs:
    #       - my_times              - list of floats. Contains the evaluation times of the training steps for each 
    #                                 epochs.
    #       - test_vals             - list of floats. Contains the objective function computed in the test steps for
    #                                 each epoch
    if not (method=='PALM' or method=='iPALM' or method=='SPRING-SARAH' or method=='iSPRING-SARAH'):
        raise ValueError('Method '+str(method)+' is unknown!')
    if test_batch_size is None:
        test_batch_size=batch_size
    if backup_dir is None:
        backup=False
    else:
        backup=True
        if not os.path.isdir(backup_dir):
            os.mkdir(backup_dir)
    if step_size is None:
        if method=='PALM':
            step_size=1.
        elif method=='iPALM':
            step_size=1.
        elif method=='SPRING-SARAH':
            step_size=0.5
        elif method=='iSPRING-SARAH':
            step_size=0.4
    if test_data is None:
        test_data=data
    if method=='SPRING-SARAH' or method=='iSPRING-SARAH':
        if sarah_p is None:
            sarah_p=20
        sarah_p_inv=1./sarah_p
    step_size=tf.constant(step_size,dtype=model.model_type)
    if method=='iSPRING-SARAH' or method=='SPRING-SARAH':
        if data is None:
            raise ValueError('Batch information is required for SPRING and iSPRING!')
        my_ds=tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0]).batch(batch_size)
    if method=='PALM' or method=='SPRING-SARAH':
        inertial_step_size=0
    if inertial_step_size is None:
        if method=='iPALM':
            inertial_step_size=1.
        elif method=='iSPRING-SARAH':
            inertial_step_size=0.5
    inertial_step_size=tf.constant(inertial_step_size,dtype=model.model_type)
    if not data is None:
        dat=data[:1]
        batch_version=True
        normal_ds=tf.data.Dataset.from_tensor_slices(data).batch(test_batch_size)
        test_ds=tf.data.Dataset.from_tensor_slices(test_data).batch(test_batch_size)
        n=data.shape[0]
    else:
        dat=None
        batch_version=False
    model(dat)
    X_vals=[]
    for i in range(model.num_blocks):
        X_vals.append(model.X[i].numpy())
    model_for_old_values=PALM_Model(X_vals,dtype=model.dtype)
    model_for_old_values.H=model.H
    if type(ensure_full)==int:
        full_steps=ensure_full
        ensure_full=False
    else:
        full_steps=np.inf    
    small_batches=batch_size>test_batch_size
    
    @tf.function
    def eval_objective():
        if batch_version:
            obj=tf.constant(0.,dtype=model.model_type)
            for batch in test_ds:
                obj+=model.objective(batch=batch)
            return obj
        else:
            return model.objective()            
    
    @tf.function
    def grad_hess_batch(mod,batch,i):
        with tf.GradientTape(persistent=True) as tape:
            val=mod(batch)
            g=tape.gradient(val,mod.X[i])
            if isinstance(g,tf.IndexedSlices):
                g2=g.values
            else:
                g2=g
            grad_sum=tf.reduce_sum(tf.multiply(g2,g2))
        h=tape.gradient(grad_sum,mod.X[i])   
        fac=0.5/tf.sqrt(grad_sum)
        h*=fac 
        g=tf.identity(g)
        h=tf.identity(h)
        return g,h

    @tf.function
    def grad_batch(mod,batch,i):
        with tf.GradientTape() as tape:
            val=mod(batch)
        g=tape.gradient(val,mod.X[i])
        return g

    def train_step_full(step,i):
        extr=inertial_step_size*(step-1.)/(step+2.)
        Xi_save=tf.identity(model.X[i])
        model.X[i].assign_sub(extr*(model_for_old_values.X[i]-model.X[i]))
        old_arg=tf.identity(model.X[i])
        if batch_version:
            grad=tf.zeros_like(model.X[i])
            hess=tf.zeros_like(model.X[i])
            eval_hess=True
            for batch in normal_ds:
                if eval_hess or not estimate_lipschitz:
                    g,h=grad_hess_batch(model,batch,i)
                    grad+=g
                    hess+=h
                    eval_hess=False
                else:
                    g=grad_batch(model,batch,i)
                    grad+=g
        else:
            grad,hess=grad_hess_batch(model,None)
        Lip=tf.sqrt(tf.reduce_sum(tf.multiply(hess,hess)))        
        if estimate_lipschitz:        
            Lip*=n*1.0/test_batch_size
        tau_i=tf.identity(Lip)
        tau_i=tf.math.multiply_no_nan(tau_i,1.-tf.cast(tf.math.is_nan(Lip),dtype=model.model_type))+1e10*tf.cast(tf.math.is_nan(Lip),dtype=model.model_type)
        model.X[i].assign(model.prox_funs[i](model.X[i]-grad/tau_i*step_size,tau_i/step_size))
        model_for_old_values.X[i].assign(Xi_save)
        return grad,old_arg
    
    def train_step_not_full(step,grads,batch,old_arg,i):
        extr=inertial_step_size*(step-1.)/(step+2.)
        Xi_save=tf.identity(model.X[i])
        model.X[i].assign_sub(extr*(model_for_old_values.X[i]-model.X[i]))
        old_arg_new=tf.identity(model.X[i])
        if small_batches:
            step_ds=tf.data.Dataset.from_tensor_slices(batch).batch(test_batch_size)
            g=tf.zeros_like(model.X[i])
            h=tf.zeros_like(model.X[i])            
            for small_batch in step_ds:
                g_b,h_b=grad_hess_batch(model,small_batch,i)
                g+=g_b
                h+=h_b
        else:
            g,h=grad_hess_batch(model,batch,i)
        Lip=tf.sqrt(tf.reduce_sum(tf.multiply(h,h)))
        tau_i=n*1.0/batch_size*tf.identity(Lip)
        tau_i=tf.math.multiply_no_nan(tau_i,1.-tf.cast(tf.math.is_nan(Lip),dtype=model.model_type))+1e10*tf.cast(tf.math.is_nan(Lip),dtype=model.model_type)
        model_for_old_values.X[i].assign(old_arg)
        if small_batches:
            g_o=tf.zeros_like(model.X[i])
            for small_batch in step_ds:
                g_o+=grad_batch(model_for_old_values,small_batch,i)
        else:
            g_o=grad_batch(model_for_old_values,batch,i)
        grad=n*1.0/batch_size*(g-g_o)+grads
        model.X[i].assign(model.prox_funs[i](model.X[i]-grad/tau_i*step_size,tau_i/step_size))
        model_for_old_values.X[i].assign(Xi_save)
        return grad,old_arg_new

    size_tensors=[tf.constant([0]),tf.constant([0,1]),tf.constant([0,1,2]),tf.constant([0,1,2,3]),tf.constant([0,1,2,3,4])]

    if precompile:
        print('precompile functions for comparing runtimes')
        grads=[None]*model.num_blocks
        old_args=[None]*model.num_blocks
        X_save=[]
        for i in range(model.num_blocks):
            X_save.append(tf.identity(model.X[i]))
        print('Compile full steps')
        for i in range(model.num_blocks):
            out=train_step_full(tf.convert_to_tensor(1,dtype=model.model_type),i)
            grads[i]=out[0]
            old_args[i]=out[1]
        if method=='SPRING-SARAH' or method=='iSPRING-SARAH':
            print('Compile stochastic steps')
            for i in range(model.num_blocks):
                train_step_not_full(tf.convert_to_tensor(1,dtype=model.model_type),grads[i],data[:batch_size],old_args[i],i)
        for i in range(model.num_blocks):
            model.X[i].assign(X_save[i])
            model_for_old_values.X[i].assign(X_save[i])
        print('precompiling finished')

    step=0
    component=0
    grads=[None]*model.num_blocks
    old_args=[None]*model.num_blocks
    print('evaluate objective')
    test_vals=[eval_objective().numpy()]
    template='Initial objective: {0:2.4f}'
    print(template.format(test_vals[0]))
    my_time=0.
    my_times=[0.]
    
    
    for epoch in range(EPOCHS):
        if batch_version:
            count=0
            if method=='PALM' or method=='iPALM':
                step+=1
                for i in range(model.num_blocks):
                    tic=time.time()
                    out=train_step_full(tf.convert_to_tensor(step,dtype=model.model_type),i)
                    toc=time.time()-tic
                    my_time+=toc
            else:
                cont=True
                while cont:
                    if steps_per_epoch==np.inf:
                        cont=False
                    for batch in my_ds:
                        #print(count)
                        if step==0:
                            step+=1
                        if component==model.num_blocks:
                            step+=1
                            component=0
                            count+=1
                            if count>=steps_per_epoch:
                                cont=False
                                break
                        if sarah_seq is None:
                            rand_num=tf.random.uniform(shape=[1],minval=0,maxval=1,dtype=tf.float32)
                        else:
                            rand_num=sarah_seq[(step-1)*model.num_blocks+component]
                        full=False
                        if step==1 or rand_num<sarah_p_inv:
                            full=True
                        if count==1 and ensure_full:
                            full=True
                        if (step-1)%full_steps==0:
                            full=True
                        tic=time.time()
                        if full:
                            print('full step')
                            out=train_step_full(tf.convert_to_tensor(step,dtype=model.model_type),component)
                        else:
                            out=train_step_not_full(tf.convert_to_tensor(step,dtype=model.model_type),grads[component],batch,old_args[component],component)
                        toc=time.time()-tic
                        my_time+=toc
                        grads[component]=out[0]
                        old_args[component]=out[1]
                        component+=1
                
        else:
            step+=1
            for i in range(model.num_blocks):
                train_step_full(tf.convert_to_tensor(step,dtype=model.model_type),i)
    
        print('evaluate objective')
        obj=eval_objective()
        template = 'Epoch {0}, Objective: {1:2.4f}, Time: {2:2.2f}'
        print(template.format(epoch+1,obj,my_time))
        if backup:
            for i in range(model.num_blocks):
                model.X[i].numpy().tofile(backup_dir+'/epoch'+str(epoch+1)+'X'+str(i))
        test_vals.append(obj.numpy())
        my_times.append(my_time)
    return my_times,test_vals



