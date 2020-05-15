This code belongs to the paper [3]. Please cite [3], if you use this code.
It contains a general implementation of PALM [1], iPALM [4], SPRING-SARAH [2] and 
iSPRING-SARAH [3] for minimizing nonconvex and nonsmooth functions of the form 
F(x_1,...,x_s)= H(x_1,...,x_s) + \sum_{i=1}^s f_i(x_i). An overview over all of these 
algorithms can be found in [3].
Convergence results are proven in the corresponding references under suitable assumptions.
In particular, it is required that H is continuously differentiable and that the f_i are
lower semicontinuous for i=1,...,s.
For questions and bug reports, please contact Johannes Hertrich (j.hertrich(at)math.tu-berlin.de).

CONTENTS:

1. REQUIREMENTS
2. USAGE
3. CLASSES AND FUNCTIONS
4. EXAMPLES
5. REFERENCES

--------------- 1. REQUIREMENTS ------------------------

The script palm_algs.py requires the Python packages Numpy and Tensorflow 2.0.0. The examples also use the
package Matplotlib.
We tested the code using the following versions of the Python packages.

Python          3.7.5
Numpy           1.17.3
Tensorflow      2.0.0
Matplotlib      3.1.1

Usually the code is also compatible with some other versions of the corresponding Python packages.

--------------- 2. USAGE -------------------------------

In this section, we give a short intruduction into the usage of this script. We do this by implementing 
iSPRING-SARAH for the problem
F(x_1,x_2)=H(x_1,x_2)+f_1(x_1)+f(x_2)
where f_1 is the characteristic function of the 1-Ball around 0 and f_2=||x_2||_1 and
H(x_1,x_2)=\sum_{i=1}^n h(x_1,x_2,y_i)  with  h(x_1,x_2,y_i)=||x_2-y_i||^2-||x_1-y_i||^2
and some a priori defined data points y_1,...,y_n.
The full script can be found in the directory with the test scripts.

First, copy the file palm_algs.py in your working directory and import its modules:


from palm_algs import *

Then we can use PALM, iPALM, SPRING-SARAH and iSPRING-SARAH algorithms by the following step-by-step implementation:

1. Prepare your problem: 

- Implement a function H(X,batch), where X is a list with the
arguments x_1,...,x_s (as numpy array). For data-based models (i.e. if H=\sum_{i=1}^n h_i(x_1,...,x_s) and
h_i(x_1,...,x_s)=h(x_1,...,x_s,y_i) for some data y_1,...,y_n), batch contains a minibatch of the data. 
For non data-based models, batch is None.

- Implement a function prox_i(x,lambda), which computes the proximal operators prox_\lambda^f_i(x) for i=1,...,s. 
The proximal operator is defined by prox_\lambda^f_i(x)=\argmin_{y}{\lambda/2 ||x-y||^2+f_i(y)}. 
In our case the proximal operators are given by the projection on the 1-Ball and the soft-shrinkage function
S_{1/lambda} with threshold 1/lambda.

- Specify inital values for x_1,..., x_s

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

- To evaluate the objective function, implement the functions f_i(x), i=1,...,s. This is optional to find
the minima of F. But if these functions are not specified, the objective value might be wrong.

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

2. Generate the PALM_model and set the parameter model.H to H and the parameter model.prox_funs to a list of
the proximal operators [prox_1,...,prox_s]. The default functions are H(X,batch)=lambda X,batch: 0 and 
prox_funs=[id,...,id], where id is the identity function.

model=PALM_Model(inits)
model.H=H
model.prox_funs=[prox_1,prox_2]
model.f=[f_1,f_2]

3. Choose the optimization method and run the algorithm by

method='iSPRING-SARAH'
optimize_PALM(model,data=data,method=method)

You can specify several further parameters. In particular, you can specify number of epochs, batch size, step size 
and the scale of the inertial parameters in iPALM or iSPRING-SARAH. For more details, see the documentation 
of optimize_PALM in Section 3. Note that for obtaining the best convergence behaviour, one has to fit these
parameters. Nethertheless, the default parameters are usually a good choice.

4. Print the results

print('X_1='+str(model.X[0].numpy())+', X_2='+str(model.X[1].numpy()))

--------------- 3. CLASSES AND FUNCTIONS ---------------

In this section, we specify the inputs of the function and class contained in palm_algs.py.

class PALM_Model:
    
Models functions of the form F(x_1,...,x_n)=H(x_1,...,x_n)+\sum_{i=1}^n f_i(x_i),
where H is continuously differentiable and the f_i are lower semicontinuous.
Inputs of the Constructor:
Required:
    initial_values          - List of numpy arrays for initialize the x_i
Optional:
    dtype                   - model type, Default: 'float32'

function optimize_PALM:

Minimizes the PALM_model using PALM/iPALM/SPRING-SARAH or iSPRING-SARAH.
Inputs:
Required:
      - model                 - PALM_Model for the objective function
Optional:
      - EPOCHS                - int. Number of epochs to optimize
                                Default value: 10
      - steps_per_epoch       - int. maximal numbers of PALM/iPALM/SPRING/iSPRING steps in one epoch
                                Default value: Infinity, that is pass the whole data set in each epoch
      - data                  - Numpy array of type model.model_type. Information to choose the minibatches. 
                                Required for SPRING and iSPRING. 
                                To run PALM/iPALM on functions, which are not data based, use data=None.
                                For SPRING and iSPRING a value not equal to None is required.
                                Default value: None
      - test_data             - Numpy array of type model.model_type. Data points to evaluate the objective   
                                function in the test step after each epoch. 
                                For test_data=None, the function uses data as test_data.
                                Default value: None
      - batch_size            - int. If data is None: No effect. Otherwise: batch_size for data driven models.
                                Default value: 1000
      - method                - String value, which declares the optimization method. Valid choices are: 'PALM', 
                                'iPALM', 'SPRING-SARAH' and 'iSPRING-SARAH'. Raises an error for other inputs.
                                Default value: 'iSPRING-SARAH'
      - inertial_step_size    - float variable. For method=='PALM' or method=='SPRING-SARAH': No effect.      
                                Otherwise: the inertial parameters in iPALM/iSPRING are chosen by 
                                inertial_step_size*(k-1)/(k+2), where k is the current step number.
                                For inertial_step_size=None the method choses 1 for PALM and iPALM, 0.5 for 
                                SPRING and 0.4 for iSPRING.
                                Default value: None
      - step_size             - float variable. The step size parameters tau are choosen by step_size*L where L 
                                is the estimated partial Lipschitz constant of H.
      - sarah_seq             - This input should be either None or a sequence of uniformly on [0,1] distributed
                                random float32-variables. The entries of sarah_seq determine if the full
                                gradient in the SARAH estimator is evaluated or not.
                                For sarah_seq=None such a sequence is created inside this function.
                                Default value: None
      - sarah_p               - float in (1,\infty). Parameter p for the sarah estimator. If sarah_p=None the 
                                method uses p=20
                                Default value: None
      - precompile            - Boolean. If precompile=True, then the functions are compiled before the time
                                measurement starts. Otherwise the functions are compiled at the first call.
                                precompiele=True makes only sense if you are interested in the runtime of the
                                algorithms without the compile time of the functions.
                                Default value: False
      - test_batch_size       - int. test_batch_size is the batch size used in the test step and in the steps
                                where the full gradient is evaluated. This does not effect the algorithm itself.
                                But it may effect the runtime. For test_batch_size=None it is set to batch_size.
                                Default value: None
      - ensure_full           - Boolean. For method=='SPRING-SARAH' or method=='iSPRING-SARAH': If ensure_full 
                                is True, we evaluate in the first step of each epoch the full gradient. We  
                                observed numerically, that this sometimes increases stability and convergence 
                                speed of SPRING and iSPRING. For PALM and iPALM: no effect.
                                Default value: False
      - estimate_lipschitz    - Boolean. If estimate_lipschitz==True, the Lipschitz constants are estimated based
                                on the first minibatch in all steps, where the full gradient is evaluated.
                                Default: True

Outputs:
      - my_times              - list of floats. Contains the evaluation times of the training steps for each 
                                epochs.
      - test_vals             - list of floats. Contains the objective function computed in the test steps for
                                each epoch

--------------- 4. EXAMPLES ----------------------------

We provide three examples for the usage of this implementation.

The first example is tutorial.py, the script described in Section 2. In this example we demonstrate the how 
to use our code. 
For more details, see Section 2.

The second example Student_t_MMs.py computes the Maximum Likelihood Estimator of Student-t mixture models. 
That is, we consider for data points x_1,....,x_n and a fixed number of components K the negative log-likelihood
function
L(alpha,nu,mu,Sigma)=-\sum_{i=1}^n\log(\sum_{k=1}^K alpha_k f(x_i|nu_k,mu_k,Sigma_k))
where f is the probability density function of the Student-t distribution. This requires that nu_k>0, \sum alpha_k=1
and that Sigma_k is a symmetric positive definite matrix for all k. Since this constraints are not lower
semicontinuous, we apply the following transformations for some eps>0:
phi_1(alpha_k)=exp(alpha_k)/(sum_{l=1}^Kexp(alpha_l)),
phi_2(nu_k)=nu_k^2+eps,
phi_3(Sigma_k)=Sigma_k^T*Sigma_K+eps*Id.
Now we have an unconstrained problem and choose for the minimization with PALM, iPALM, SPRING-SARAH and iSPRING-SARAH:
H(alpha,nu,mu,Sigma)=L(phi_1(alpha),phi_2(nu),mu,phi_3(Sigma)
and f_1=f_2=f_3=f_4=0. Thus PALM becomes basically a block gradient descent algorithm.
For more detailed description of this example, we refer to [3]. 

The third example computes the sparse PCA of some data points. That is, to minimize
min_{X,Y} ||A-XY||_F^2+||X||_1+||Y||_1,
where A is a n x d matrix, X is a n x r and Y is a r x d matrix and ||X||_1 denotes the 1-Norm of the vectorized
matrix X.
For more details and further references on the sparse PCA we refer to [2] and [3].
We implement the sparse PCA by minimizing
H(X,Y)+f_1(X)+f_2(Y)
with H(X,Y)=\sum_{i=1}^n h_i(X,Y), where h_i(X,Y)=A_i-X_iY, where A_i and X_i denote the i-th line of A and X.
Note that the gradient of h_i with respect to X is sparse. Since the general framework in palm_algs.py allocates 
nethertheless the full gradient matrix wrt. to X, a more special implementation of SPRING-SARAH and iSPRING-SARAH
yields a better convergence behaviour. Thus, we provide two implementations of the sparse PCA. The first one,
sparse_PCA_palm_algs.py uses the framework palm_algs. For the comparison of PALM, iPALM, SPRING-SARAH and
iSPRING-SARAH in [3] we use an implementation by scratch in sparse_PCA_compare.py. Here we do not compile the
Tensorflow functions to get a better comparability of the algorithms.

--------------- 5. REFERENCES --------------------------

[1] J. Bolte, S. Sabach, and M. Teboulle.
Proximal alternating linearized minimization for nonconvex and nonsmooth problems.
Mathematical Programming, 146(1-2, Ser. A):459–494, 2014.

[2] D. Driggs, J. Tang, J. Liang, M. Davies, and C.-B. Schönlieb.
SPRING: A fast stochastic proximal alternating method for non-smooth non-convex optimization.
ArXiv preprint arXiv:2002.12266, 2020.

[3] J. Hertrich and G. Steidl. 
Inertial stochastic palm and its application for learning Student-t mixture models. 
ArXiv preprint arXiv:2005.02204, 2020.

[4] T. Pock and S. Sabach.
Inertial proximal alternating linearized minimization (iPALM) for nonconvex and nonsmooth problems.
SIAM Journal on Imaging Sciences, 9(4):1756–1787, 2016.

