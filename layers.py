import tensorflow as tf
import morpholayers.layers as ml
#from functions import *


def count(images):
    """Plot images in one row."""
    tmp = np.sum(images)
    #print("sum = :", tmp)
    return tmp

def update_dilation(last,new,mask):
     return [new, geodesic_dilation_step([new, mask]), mask]

@tf.function
def condition_equal(last,new,image):
    return tf.math.logical_not(tf.reduce_all(tf.math.equal(last, new)))

@tf.function
def geodesic_dilation_step(X):
    """
    1 step of reconstruction by dilation
    :X tensor: X[0] is the Mask and X[1] is the Image
    :param steps: number of steps (by default NUM_ITER_REC)
    :Example:
    >>>Lambda(geodesic_dilation_step, name="reconstruction")([Mask,Image])
    """
    # perform a geodesic dilation with X[0] as marker, and X[1] as mask
    return tf.keras.layers.Minimum()([tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(X[0]),X[1]])

@tf.function
def geodesic_dilation(X,steps=None):
    """
    Full reconstruction by dilation if steps=None, else
    K steps reconstruction by dilation
    :X tensor: X[0] is the Mask and X[1] is the Image
    :param steps: number of steps (by default NUM_ITER_REC)
    :Example:
    >>>Lambda(geodesic_dilation, name="reconstruction")([Mask,Image])
    """
    rec = X[0]
    #Full reconstruction is steps==None by dilation, else: partial reconstruction
    rec = geodesic_dilation_step([rec, X[1]])
    _, rec,_=tf.while_loop(condition_equal,
                            update_dilation,
                            [X[0], rec, X[1]],
                            maximum_iterations=steps)
    return rec


def reconstruction_dilation(X):
    """
    Full geodesic reconstruction by dilation, reaching idempotence
    :X tensor: X[0] is the Mask and X[1] is the Image
    :param steps: number of steps (by default NUM_ITER_REC)
    :Example:
    >>>Lambda(reconstruction_dilation, name="reconstruction")([Mask,Image])
    """
    return geodesic_dilation(X, steps=None)



@tf.function
def count_h_maxima(xinput):
    epsilon = 1e-5
    x=xinput[0]
    x=tf.stop_gradient(x)
    h=xinput[1]
    h=tf.expand_dims(tf.expand_dims(h,axis=-1),axis=-1)

    xh=reconstruction_dilation([x-h,x])
    Rmax=(xh-reconstruction_dilation([xh-epsilon,xh]))>0
    Rmax = tf.cast(Rmax, tf.dtypes.float32)
    U=Sampling()(Rmax)
    M=tf.keras.layers.Minimum()([U,Rmax])
    R=tf.keras.layers.Lambda(geodesic_dilation)([M,Rmax])
    Detection=tf.cast(U==R,tf.float32)
    CC=tf.math.reduce_sum(Detection,axis=[1,2])

    # CC should be zero if xh is constant
    # or equivalently if RMax is constant equal to one
    LAMDA = 100.
    CC_ = tf.math.minimum(CC, LAMDA*(tf.reduce_max(Rmax) - tf.reduce_min(Rmax))*CC)
    return CC_

@tf.function
def count_h_maxima_and_rec(xinput):
    epsilon = 1e-5
    x=xinput[0]
    x=tf.stop_gradient(x)
    h=xinput[1]
    h=tf.expand_dims(tf.expand_dims(h,axis=-1),axis=-1)

    xh=reconstruction_dilation([x-h,x])
    Rmax=(xh-reconstruction_dilation([xh-epsilon,xh]))>0
    Rmax = tf.cast(Rmax, tf.dtypes.float32)
    U=Sampling()(Rmax)
    M=tf.keras.layers.Minimum()([U,Rmax])
    R=tf.keras.layers.Lambda(geodesic_dilation)([M,Rmax])
    Detection=tf.cast(U==R,tf.float32)
    CC=tf.math.reduce_sum(Detection,axis=[1,2])

    # CC should be zero if xh is constant
    # or equivalently if RMax is constant equal to one
    LAMDA = 100.
    CC_ = tf.math.minimum(CC, LAMDA*(tf.reduce_max(Rmax) - tf.reduce_min(Rmax))*CC)
    return CC_, xh, Rmax



# @tf.custom_gradient
# def custom_h_rec_and_exact_cc_minus_one(xinput):
#     def grad(upstream):
#         return_grad = -upstream
#         return upstream, return_grad
#     CC = count_h_maxima(xinput)
#     return CC, grad



# @tf.custom_gradient
# def custom_h_rec_and_exact_cc_N(xinput):
#     epsilon = 1e-5
#     N=50
#     def grad(upstream):
#         x=xinput[0]
#         h=xinput[1]
#         h=tf.expand_dims(tf.expand_dims(h,axis=-1),axis=-1)
#         noise_amp = 0.2*h
#         #noise=tf.random.uniform(shape=tf.shape(x), minval=-noise_amp, maxval=noise_amp)
#         noise=noise_amp*tf.random.normal(shape=tf.shape(x))
#         x=tf.add(x,noise)
#         x=tf.stop_gradient(x)
#         xh=reconstruction_dilation([x-h,x])
#         Rmax=(xh-reconstruction_dilation([xh-epsilon,xh]))>0
#         Rmax = tf.cast(Rmax, tf.dtypes.float32)
#         dxh_dh = -Rmax
#         rho = (x-xh)/(h)
#         #rho_N_1 = rho**(N-1)
#         rho_N_1 = tf.math.maximum(1-(N-1)*(1-rho), 0)
#         InPrime = -N*rho_N_1*(rho+dxh_dh)/h
#         deriv = tf.reduce_sum(InPrime, axis=[1,2])
#         return_grad = upstream*deriv    
#         return upstream, return_grad

#     CC = count_h_maxima(xinput)
#     return CC, grad


class ExpandtoImageLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.expand_dims(tf.expand_dims(x,axis=-1),axis=-1)


class countingLayer(tf.keras.layers.Layer):
        def __init__(self, bp_mode, N=50, **kwargs):
            super().__init__(**kwargs)
            self.bp_mode = bp_mode
            self.N = N

        @tf.custom_gradient
        def custom_exact_cc_minus_one(self,xinput):
            def grad(upstream):
                return_grad = -upstream
                return upstream, return_grad
            CC = count_h_maxima(xinput)
            return CC, grad

        @tf.custom_gradient
        def custom_exact_cc_N(self,xinput):
            epsilon = 1e-5
            N=self.N
            def grad(upstream):
                x=xinput[0]
                h=xinput[1]
                h=tf.expand_dims(tf.expand_dims(h,axis=-1),axis=-1)
                noise_amp = 0.2*h
                noise=noise_amp*tf.random.normal(shape=tf.shape(x))
                x=tf.add(x,noise)
                x=tf.stop_gradient(x)
                xh=reconstruction_dilation([x-h,x])
                Rmax=(xh-reconstruction_dilation([xh-epsilon,xh]))>0
                Rmax = tf.cast(Rmax, tf.dtypes.float32)
                dxh_dh = -Rmax
                rho = (x-xh)/(h)
                #rho_N_1 = rho**(N-1)
                rho_N_1 = tf.math.maximum(1-(N-1)*(1-rho), 0)
                InPrime = -N*rho_N_1*(rho+dxh_dh)/h
                deriv = tf.reduce_sum(InPrime, axis=[1,2])
                return_grad = upstream*deriv    
                return upstream, return_grad
            CC = count_h_maxima(xinput)
            return CC, grad

        def call(self, xinput):
            if self.bp_mode == "minus_one":
                return self.custom_exact_cc_minus_one(xinput)
            elif self.bp_mode == "Puissance_N":
                return self.custom_exact_cc_N(xinput)
            else:
                raise NotImplementedError


class countAndRecLayer(tf.keras.layers.Layer):
    def __init__(self, bp_mode, N=50, **kwargs):
        super().__init__(**kwargs)
        self.bp_mode = bp_mode
        self.N = N

    @tf.custom_gradient
    def custom_exact_cc_rec_minus_one(self,xinput):
        
        CC, Xh, Rmax = count_h_maxima_and_rec(xinput)
        
        def grad(*upstream):
            upstream_count = upstream[0]
            upstream_rec = upstream[1]
            grad_rec = tf.reduce_sum(-Rmax*upstream_rec, axis=[1,2])
            grad_count = -upstream_count
            return [grad_count, grad_rec]

        return [CC, Xh], grad
    

    @tf.custom_gradient
    def custom_exact_cc_rec_N(self,xinput):
        epsilon = 1e-5
        N=self.N
        CC, Xh, Rmax = count_h_maxima_and_rec(xinput)
        
        def grad(*upstream):
            upstream_count = upstream[0]
            upstream_rec = upstream[1]
            grad_rec = tf.reduce_sum(-Rmax*upstream_rec, axis=[1,2])
            x=xinput[0]
            h=xinput[1]
            h=tf.expand_dims(tf.expand_dims(h,axis=-1),axis=-1)
            noise_amp = 0.2*h
            noise=noise_amp*tf.random.normal(shape=tf.shape(x))
            x=tf.add(x,noise)
            x=tf.stop_gradient(x)
            xh=reconstruction_dilation([x-h,x])
            Rmax_noise=(xh-reconstruction_dilation([xh-epsilon,xh]))>0
            Rmax_noise = tf.cast(Rmax_noise, tf.dtypes.float32)
            dxh_dh = -Rmax_noise
            rho = (x-xh)/(h)
            #rho_N_1 = rho**(N-1)
            rho_N_1 = tf.math.maximum(1-(N-1)*(1-rho), 0)
            InPrime = -N*rho_N_1*(rho+dxh_dh)/h
            deriv = tf.reduce_sum(InPrime, axis=[1,2])
            grad_count = upstream_count*deriv    
            return [grad_count, grad_rec]

        return [CC, Xh], grad

    def call(self, xinput):
        if self.bp_mode == "minus_one":
            return self.custom_exact_cc_rec_minus_one(xinput)
        elif self.bp_mode == "Puissance_N":
            return self.custom_exact_cc_rec_N(xinput)
        else:
            raise NotImplementedError
        
    
            
class Sampling(tf.keras.layers.Layer):
    """Sampling Random Uniform."""

    def call(self, inputs):
        dim = tf.shape(inputs)
        epsilon = tf.keras.backend.random_uniform(shape=(dim))/100
        return epsilon




# class HrecExactCC(tf.keras.layers.Layer):
#         def __init__(self, bp_mode, N=50, **kwargs):
#             super().__init__(**kwargs)
#             self.bp_mode = bp_mode

#         def call(self, xinput):
#             if self.bp_mode == "minus_one":
#                 return custom_h_rec_and_exact_cc_minus_one(xinput)
#             elif self.bp_mode == "Puissance_N":
#                 return custom_h_rec_and_exact_cc_N(xinput)
#             else:
#                 raise NotImplementedError

