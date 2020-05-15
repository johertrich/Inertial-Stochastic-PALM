from palm_algs import *
import matplotlib.pyplot as plt

lambda_1=1
lambda_2=1
n=1000000
d=30
r=5

A=50*tf.random.uniform(shape=[n,d])
# required functions
def H(X,batch):
    X_batch=tf.gather(X[0],batch)
    A_batch=tf.gather(A,batch)
    inner=A_batch-tf.matmul(X_batch,X[1])
    return tf.reduce_sum(inner**2)

def f1(X):
    return lambda_1*tf.reduce_sum(tf.abs(X))

def prox_1(arg,lam):
    soft_thresh=lambda_1/lam
    return tf.math.sign(arg)*tf.math.maximum(0.,tf.math.abs(arg)-soft_thresh)

def f2(X):
    return lambda_2*tf.reduce_sum(tf.abs(X))

def prox_2(arg,lam):
    soft_thresh=lambda_2/lam
    return tf.math.sign(arg)*tf.math.maximum(0.,tf.math.abs(arg)-soft_thresh)

batch_size=10000
epch=20
steps_per_epch=10

X_init=20*tf.random.uniform(shape=[n,r]).numpy()
Y_init=20*tf.random.uniform(shape=[r,d]).numpy()

# warm start and initializiation
model=PALM_Model([X_init,Y_init])
model.H=H
model.prox_funs[0]=prox_1
model.prox_funs[1]=prox_2
model.f[0]=f1
model.f[1]=f2
optimize_PALM(model,data=np.array(range(n)),batch_size=batch_size,method='PALM',EPOCHS=2)
model2=PALM_Model([model.X[0].numpy(),model.X[1].numpy()])
model2.H=H
model2.prox_funs[0]=prox_1
model2.prox_funs[1]=prox_2
model2.f[0]=f1
model2.f[1]=f2
model3=PALM_Model([model.X[0].numpy(),model.X[1].numpy()])
model3.H=H
model3.prox_funs[0]=prox_1
model3.prox_funs[1]=prox_2
model3.f[0]=f1
model3.f[1]=f2
model4=PALM_Model([model.X[0].numpy(),model.X[1].numpy()])
model4.H=H
model4.prox_funs[0]=prox_1
model4.prox_funs[1]=prox_2
model4.f[0]=f1
model4.f[1]=f2

# run algorithms
sarah_seq=tf.random.uniform(shape=[epch*steps_per_epch*4+100],minval=0,maxval=1,dtype=tf.float32)
ispring=optimize_PALM(model,data=np.array(range(n)),batch_size=batch_size,method='iSPRING-SARAH',EPOCHS=epch,step_size=0.4,steps_per_epoch=steps_per_epch,precompile=True,sarah_seq=sarah_seq)
spring=optimize_PALM(model2,data=np.array(range(n)),batch_size=batch_size,method='SPRING-SARAH',EPOCHS=epch,step_size=0.5,steps_per_epoch=steps_per_epch,precompile=True,sarah_seq=sarah_seq)
palm=optimize_PALM(model3,data=np.array(range(n)),batch_size=batch_size,method='PALM',EPOCHS=epch,precompile=True)
ipalm=optimize_PALM(model4,data=np.array(range(n)),batch_size=batch_size,method='iPALM',EPOCHS=epch,precompile=True)


# Generate plots
fig=plt.figure()
plt.plot(palm[1],'-',c='red')
plt.plot(ipalm[1],'--',c='green')
plt.plot(spring[1],'-.',c='black')
plt.plot(ispring[1],':',c='blue')
plt.legend(['PALM','iPALM','SPRING-SARAH','iSPRING-SARAH'])
fig.savefig('sparse_PCA_palm_algs.png',dpi=1200)
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
fig.savefig('sparse_PCA_palm_algs_times.png',dpi=1200)
plt.close(fig)
