# Inertial Stochastic PALM (iSPALM) and Applications in Machine Learning

This code belongs to the paper [3]. Please cite [3], if you use this code.

It contains a general implementation of PALM [1], iPALM [4], SPRING-SARAH [2] and 
iSPALM-SARAH [3] for minimizing nonconvex and nonsmooth functions of the form 
F(x_1,...,x_s)= H(x_1,...,x_s) + \sum_{i=1}^s f_i(x_i). An overview over all of these 
algorithms can be found in [3].
Convergence results are proven in the corresponding references under suitable assumptions.
In particular, it is required that H is continuously differentiable and that the f_i are
lower semicontinuous for i=1,...,s.
For questions and bug reports, please contact Johannes Hertrich (j.hertrich(at)math.tu-berlin.de).

## CONTENTS

1. REQUIREMENTS
2. USAGE
3. CLASSES AND FUNCTIONS
4. EXAMPLES
5. REFERENCES


## 1. REQUIREMENTS

The script palm_algs.py requires the Python packages Numpy and Tensorflow 2.0.0. The examples also use the
package Matplotlib.
We tested the code using the following versions of the Python packages.

Python          3.7.5
Numpy           1.17.3
Tensorflow      2.0.0
Matplotlib      3.1.1

Usually the code is also compatible with some other versions of the corresponding Python packages.

## 2. USAGE

In this section, we give a short intruduction into the usage of this script. We do this by implementing 
iSPALM-SARAH for the problem

> F(x_1,x_2)=H(x_1,x_2)+f_1(x_1)+f(x_2),

where f_1 is the characteristic function of the 1-Ball around 0 and f_2=||x_2||\_1 and

> H(x_1,x_2)=\sum\_{i=1}^n h(x_1,x_2,y_i)  with  h(x_1,x_2,y_i)=||x_2-y_i||^2-||x_1-y_i||^2

and some a priori defined data points y_1,...,y_n.
The full script can be found in the directory with the test scripts.

1.) First, copy the file `palm_algs.py` in your working directory and import its modules:

```
from palm_algs import *
```

Then we can use PALM, iPALM, SPRING-SARAH and iSPALM-SARAH algorithms by the following step-by-step implementation:

2.) Prepare your problem: 

- Implement a function `H(X,batch)`, where X is a list with the
arguments `x_1,...,x_s` (as `numpy` array). For data-based models (i.e. if H=\sum\_{i=1}^n h_i(x_1,...,x_s) and
h_i(x_1,...,x_s)=h(x_1,...,x_s,y_i) for some data y_1,...,y_n), `batch` contains a minibatch of the data. 
For non data-based models, `batch` is `None`.

- Implement a function `prox_i(x,lambda)`, which computes the proximal operators prox\_\lambda^f_i(x) for i=1,...,s. 
The proximal operator is defined by prox\_\lambda^f_i(x)=\argmin\_{y}{\lambda/2 ||x-y||^2+f_i(y)}. 
In our case the proximal operators are given by the projection on the 1-Ball and the soft-shrinkage function
S\_{1/lambda} with threshold 1/lambda.
*NOTE THAT THE FUNCTIONS H AND prox_i SHOULD BE COMPILABLE TO A TENSORFLOW GRAPH!!!*

- Specify initial values for `x_1,..., x_s`

```
def H(X,batch):
    diffs1=tf.add(batch,-X[0])
    diffs2=tf.add(batch,-X[1])
    return tf.reduce_sum(diffs2**2)-tf.reduce_sum(diffs1**2)

def prox_1(x,lam):
    return x/tf.sqrt(tf.reduce_sum(x**2))

def prox_2(x,lam):
    return tf.multiply(tf.math.sign(x),tf.math.maximum(tf.math.abs(x)-1/lam,0))

d=5
inits=[np.zeros(d),np.zeros(d)]
```

- To evaluate the objective function, implement the functions `f_i(x)`, i=1,...,s. This is optional to find
the minimum of F. But if these functions are not specified, the objective value might be wrong.

```
def f_1(x):
    my_norm=tf.math.sqrt(tf.reduce_sum(x**2))
    if my_norm>1.:
        a=tf.constant(np.inf,dtype=tf.float32)
    else:
        a=tf.constant(0,dtype=tf.float32)
    return a

def f_2(x):
    return tf.reduce_sum(tf.math.abs(x))

- For data based models, you should also prepare your data.

n=10000
data=np.random.normal(loc=0.5,size=[n,d])
```

3.) Generate the `PALM_model` and set the parameter `model.H` to H and the parameter `model.prox_funs` to a list of
the proximal operators `[prox_1,...,prox_s]`. The default functions are `H(X,batch)=lambda X,batch: 0` and 
`prox_funs=[id,...,id]`, where `id` is the identity function.

```
model=PALM_Model(inits)
model.H=H
model.prox_funs=[prox_1,prox_2]
model.f=[f_1,f_2]
```

4.) Choose the optimization method, create the optimizer and run the algorithm by

```
method='iSPALM-SARAH'
optimizer=PALM_Optimizer(model,data=data,method=method)
optimizer.optimize(EPOCHS=10)
```

You can specify several further parameters. In particular, you can specify number of epochs, batch size, step size 
and the scale of the inertial parameters in iPALM or iSPALM-SARAH. For more details, see the documentation 
of `PALM_optimizer` in Section 3. Note that for obtaining the best convergence behaviour, one has to fit these
parameters. Nevertheless, the default parameters are usually a good choice.

4.) Print the results

```
print('X_1='+str(model.X[0].numpy())+', X_2='+str(model.X[1].numpy()))
```

## 3. CLASSES AND FUNCTIONS

In this section, we specify the inputs of the function and class contained in palm_algs.py.

### class PALM_Model:
    
Models functions of the form F(x_1,...,x_n)=H(x_1,...,x_n)+\sum_{i=1}^n f_i(x_i),
where H is continuously differentiable and the f_i are lower semicontinuous.  
**Inputs of the Constructor:**  
**Required:**  

- **initial_values**          - List of numpy arrays for initialize the x_i  

**Optional:**  

- **dtype**                   - model type, Default: 'float32'  

### class PALM_Optimizer:

Optimizer class for functions implemented as PALM_Model.  
**Inputs for the Constructor:**  
**Required:**  

- model                 - PALM_Model for the objective function

**Optional:**

- **steps_per_epoch**       - int. maximal numbers of PALM/iPALM/SPRING/iSPALM steps in one epoch
                                Default value: Infinity, that is pass the whole data set in each epoch
- **data**                  - Numpy array of type model.model_type. Information to choose the minibatches. 
                                Required for SPRING and iSPALM. 
                                To run PALM/iPALM on functions, which are not data based, use data=None.
                                For SPRING and iSPALM a value not equal to None is required.
                                Default value: None
- **test_data**             - Numpy array of type model.model_type. Data points to evaluate the objective   
                                function in the test step after each epoch. 
                                For test_data=None, the function uses data as test_data.
                                Default value: None
- **batch_size**            - int. If data is None: No effect. Otherwise: batch_size for data driven models.
                                Default value: 1000
- **method**                - String value, which declares the optimization method. Valid choices are: 'PALM', 
                                'iPALM', 'SPRING-SARAH' and 'iSPALM-SARAH'. Raises an error for other inputs.
                                Default value: 'iSPALM-SARAH'
- **inertial_step_size**    - float variable. For method=='PALM' or method=='SPRING-SARAH': No effect.      
                                Otherwise: the inertial parameters in iPALM/iSPALM are chosen by 
                                inertial_step_size*(k-1)/(k+2), where k is the current step number.
                                For inertial_step_size=None the method choses 1 for PALM and iPALM, 0.5 for 
                                SPRING and 0.4 for iSPALM.
                                Default value: None
- **step_size**             - float variable. The step size parameters tau are choosen by step_size*L where L 
                                is the estimated partial Lipschitz constant of H.
- **sarah_seq**             - This input should be either None or a sequence of uniformly on [0,1] distributed
                                random float32-variables. The entries of sarah_seq determine if the full
                                gradient in the SARAH estimator is evaluated or not.
                                For sarah_seq=None such a sequence is created inside this function.
                                Default value: None
- **sarah_p**               - float in (1,\infty). Parameter p for the sarah estimator. If sarah_p=None the 
                                method uses p=20
                                Default value: None
- **test_batch_size**       - int. test_batch_size is the batch size used in the test step and in the steps
                                where the full gradient is evaluated. This does not effect the algorithm itself.
                               But it may effect the runtime. For test_batch_size=None it is set to batch_size.
                                If test_batch_size<batch_size and method=SPRING-SARAH or method=iSPALM-SARAH,
                                then also in the steps, where not the full gradient is evaluated only batches
                                of size test_batch_size are passed through the function H.
                                Default value: None
- **ensure_full**           - Boolean or int. For method=='SPRING-SARAH' or method=='iSPALM-SARAH': If
                                ensure_full is True, we evaluate in the first step of each epoch the full
                                gradient. We observed numerically, that this sometimes increases stability and
                                convergence speed of SPRING and iSPALM. For PALM and iPALM: no effect.
                                If a integer value p is provided, every p-th step is forced to be a full step
                                Default value: False
- **estimate_lipschitz**    - Boolean. If estimate_lipschitz==True, the Lipschitz constants are estimated based
                               on the first minibatch in all steps, where the full gradient is evaluated.
                                Default: True
- **backup_dir**            - String or None. If a String is provided, the variables X[i] are saved after
                                every epoch. The weights are not saved if backup_dir is None.
                                Default: 'backup'
- **mute**                  - Boolean. For mute=True the evaluation of the objective function and all prints
                                will be suppressed.
                                Default: False

**Provides the following functions:**

- **evaluate_objective**    - evaluates the objective function for the current parameters
                                Inputs: None
                                Outputs: None
- **precompile**            - compiles parts of the functions to tensorflow graphs to compare runtimes
                                Inputs: None
                                Outputs: None
- **exec_epoch**            - executes one epoch of the algorithm
                                Inputs: None
                                Outputs: None
- **optimize**              - executes a fixed number of epoch  
  **Inputs:**  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **EPOCHS**            - Number of epochs.  
  **Outputs:**  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    **my_times**          - Cummulated execution time of the epochs  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     **test_vals**         - Objective function on the test set.
                                                             If test_data is None, then the objective function
                                                             on the data set. Only returned if mute==False.  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **train_vals**        - Objective function on the train set. 
                                                             Only returned if mute==False and test_data is not None.
 
**The following functions should not be called directly:**

- **train_step_full**       - performs one step of the algorithm, where the full gradient is evaluated
- **train_step_not_full**   - performs one step of the algorithm, where not the full gradient is evaluated


## 4. EXAMPLES

We provide three examples for the usage of this implementation.

The **first example** is tutorial.py, the script described in Section 2. In this example we demonstrate the how 
to use our code. 
For more details, see Section 2.

The **second example** Student_t_MMs.py computes the Maximum Likelihood Estimator of Student-t mixture models. 
That is, we consider for data points x_1,....,x_n and a fixed number of components K the negative log-likelihood
function
L(alpha,nu,mu,Sigma)=-\sum_{i=1}^n\log(\sum_{k=1}^K alpha_k f(x_i|nu_k,mu_k,Sigma_k))
where f is the probability density function of the Student-t distribution. This requires that nu_k>0, \sum alpha_k=1
and that Sigma_k is a symmetric positive definite matrix for all k. Since this constraints are not lower
semicontinuous, we apply the following transformations for some eps>0:
phi_1(alpha_k)=exp(alpha_k)/(sum_{l=1}^Kexp(alpha_l)),
phi_2(nu_k)=nu_k^2+eps,
phi_3(Sigma_k)=Sigma_k^T*Sigma_K+eps*Id.
Now we have an unconstrained problem and choose for the minimization with PALM, iPALM, SPRING-SARAH and iSPALM-SARAH:
H(alpha,nu,mu,Sigma)=L(phi_1(alpha),phi_2(nu),mu,phi_3(Sigma)
and f_1=f_2=f_3=f_4=0. Thus PALM becomes basically a block gradient descent algorithm.
For more detailed description of this example, we refer to [3]. 

The **third example** is the training of a proximal neural network PNN. These networks were proposed in  
> &nbsp; M. Hasannasab, J. Hertrich, S.Neumayer, G. Plonka, S. Setzer and G. Steidl  
> &nbsp; Parseval Proximal Neural Networks, 2020.  
> &nbsp; Journal of Fourier Analysis and Applications, vol 26, no. 59.  

and are basically neural networks with some orthogonality constraints on the weight matrices.
For more details and further references on PNNs we refer to [3].
Our objective functional consists of  

> F(w)=H(w)+f(w)  

where H(w) is the loss functions for the weights w and f is the characteristic function of the set of feasible parameters. 

## 5. REFERENCES

[1] J. Bolte, S. Sabach, and M. Teboulle.  
Proximal alternating linearized minimization for nonconvex and nonsmooth problems.  
Mathematical Programming, 146(1-2, Ser. A):459–494, 2014.

[2] D. Driggs, J. Tang, J. Liang, M. Davies, and C.-B. Schönlieb.  
SPRING: A fast stochastic proximal alternating method for non-smooth non-convex optimization.  
ArXiv preprint arXiv:2002.12266, 2020.

[3] J. Hertrich and G. Steidl.  
Inertial Stochastic PALM (iSPALM) and Applications in Machine Learning.  
ArXiv preprint arXiv:2005.02204, 2020.

[4] T. Pock and S. Sabach.  
Inertial proximal alternating linearized minimization (iPALM) for nonconvex and nonsmooth problems.  
SIAM Journal on Imaging Sciences, 9(4):1756–1787, 2016.

