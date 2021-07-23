import numpy as np
import math as mt

class Adam(object):
    def __init__(self, lrn_rate, beta1, beta2, eps):
        self.lrn_rate = lrn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.idx = 0
        self.m = 0.0 # 1st order
        self.v = 0.0 # 2nd order

    def update(self, x, g):
        self.idx += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g ** 2
        m_adj = self.m / (1.0 - np.power(self.beta1, self.idx))
        v_adj = self.v / (1.0 - np.power(self.beta2, self.idx))
        x_new = x - self.lrn_rate * m_adj / np.sqrt(v_adj + self.eps)
        return x_new


class AdaBelief(object):
    def __init__(self, lrn_rate, beta1, beta2, eps):
        self.lrn_rate = lrn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.idx = 0
        self.m = 0.0 # 1st order
        self.v = 0.0 # 2nd order

    def update(self, x, g):
        self.idx += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g - self.m) ** 2 + self.eps
        m_adj = self.m / (1.0 - np.power(self.beta1, self.idx))
        v_adj = self.v / (1.0 - np.power(self.beta2, self.idx))
        x_new = x - self.lrn_rate * m_adj / np.sqrt(v_adj + self.eps)
        return x_new

class diffGrad(object):
    def __init__(self, lrn_rate, beta1, beta2, eps):
        self.lrn_rate = lrn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.idx = 0
        self.m = 0.0 # 1st order
        self.v = 0.0 # 2nd order
        self.g_prev = 0.0

    def update(self, x, g):
        self.idx += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g ** 2
        m_adj = self.m / (1.0 - np.power(self.beta1, self.idx))
        v_adj = self.v / (1.0 - np.power(self.beta2, self.idx))
        dfc = 1.0 / (1.0 + np.exp(-np.abs(self.g_prev - g)))
        x_new = x - self.lrn_rate * m_adj * dfc / (np.sqrt(v_adj) + self.eps)
        self.g_prev = g
        return x_new




class AngularGradCos(object):
    def __init__(self, lrn_rate, beta1, beta2, eps):
        self.lrn_rate = lrn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.idx = 0
        self.m = 0.0 # 1st order
        self.v = 0.0 # 2nd order
        self.g_prev = 0.0
        self.min = 0.0

    def update(self, x, g):
        self.idx += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g ** 2
        m_adj = self.m / (1.0 - np.power(self.beta1, self.idx))
        v_adj = self.v / (1.0 - np.power(self.beta2, self.idx))

        tan_theta = abs((self.g_prev - g) / (1 + self.g_prev * g))
        cos_theta = 1 / np.sqrt(1 + tan_theta**2)
        angle = np.arctan(tan_theta) * (180 / 3.141592653589793238)
     
        if angle > self.min:
            self.min = angle
            diff = abs(self.g_prev - g)
            final_cos_theta = cos_theta
        else:
            self.min = angle
            diff = abs(self.g_prev - g)
            final_cos_theta = cos_theta

        dfc = 1.0 / (1.0 + np.exp(final_cos_theta))
        x_new = x - self.lrn_rate * m_adj * dfc / (np.sqrt(v_adj) + self.eps)
        self.g_prev = g
        return x_new




class AngularGradTan(object):
    def __init__(self, lrn_rate, beta1, beta2, eps):
        self.lrn_rate = lrn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.idx = 0
        self.m = 0.0 # 1st order
        self.v = 0.0 # 2nd order
        self.g_prev = 0.0
        self.min = 0.0

    def update(self, x, g):
        self.idx += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g ** 2
        m_adj = self.m / (1.0 - np.power(self.beta1, self.idx))
        v_adj = self.v / (1.0 - np.power(self.beta2, self.idx))

        tan_theta = abs((self.g_prev - g) / (1 + self.g_prev * g))
        cos_theta = 1 / np.sqrt(1 + tan_theta**2)
        angle = np.arctan(tan_theta) * (180 / 3.141592653589793238)
     
        if angle > self.min:
            self.min = angle
            diff = abs(self.g_prev - g)
            final_tan_theta = tan_theta
        else:
            self.min = angle
            diff = abs(self.g_prev - g)
            final_tan_theta = tan_theta

        dfc = 1.0 / (1.0 + np.exp(final_tan_theta))
        x_new = x - self.lrn_rate * m_adj * dfc / (np.sqrt(v_adj) + self.eps)
        self.g_prev = g
        return x_new


class SGDM(object):
    def __init__(self, lrn_rate, beta1, eps):
        self.lrn_rate = lrn_rate
        self.beta1 = beta1
        self.eps = eps
        self.idx = 0
        self.m = 0.0 # 1st order
        
    def update(self, x, g):
        self.idx += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        m_adj = self.m / (1.0 - np.power(self.beta1, self.idx))
        x_new = x - self.lrn_rate * m_adj
        return x_new




#class AngularGradCos(object):
    #def __init__(self, lrn_rate, beta1, beta2, eps):
        #self.lrn_rate = lrn_rate
        #self.beta1 = beta1
        #self.beta2 = beta2
        #self.eps = eps
        #self.idx = 0
        #self.m = 0.0 # 1st order
        #self.v = 0.0 # 2nd order
        #self.g_prev = 0.0
        #self.min = 360.0

    #def update(self, x, g):
        #self.idx += 1
        #self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        #self.v = self.beta2 * self.v + (1.0 - self.beta2) * g ** 2
        #m_adj = self.m / (1.0 - np.power(self.beta1, self.idx))
        #v_adj = self.v / (1.0 - np.power(self.beta2, self.idx))

        #tan_theta = abs((self.g_prev - g) / (1 + self.g_prev * g))
        #cos_theta = 1 / np.sqrt(1 + tan_theta ** 2)

        #angle = np.arctan(tan_theta) * (180 / 3.141592653589793238)
    
        #if angle < self.min:
            #self.min = angle
            #final_cos_theta = cos_theta
        #else:
            #final_cos_theta = mt.cos(self.min)

        #dfc = np.tanh(abs(final_cos_theta)) * 0.5 +0.5
        #x_new = x - self.lrn_rate * m_adj * dfc / (np.sqrt(v_adj) + self.eps)
        #self.g_prev = g
        #return x_new




#class AngularGradTan(object):
    #def __init__(self, lrn_rate, beta1, beta2, eps):
        #self.lrn_rate = lrn_rate
        #self.beta1 = beta1
        #self.beta2 = beta2
        #self.eps = eps
        #self.idx = 0
        #self.m = 0.0 # 1st order
        #self.v = 0.0 # 2nd order
        #self.g_prev = 0.0
        #self.min = 361.0

    #def update(self, x, g):
        #self.idx += 1
        #self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        #self.v = self.beta2 * self.v + (1.0 - self.beta2) * g ** 2
        #m_adj = self.m / (1.0 - np.power(self.beta1, self.idx))
        #v_adj = self.v / (1.0 - np.power(self.beta2, self.idx))

        #tan_theta = abs((self.g_prev - g) / (1 + self.g_prev * g))
        #angle = np.arctan(tan_theta) * (180 / 3.141592653589793238)
     
        #if angle > self.min:
            #self.min = angle
            #final_tan_theta = tan_theta
        #else:
            #final_tan_theta = mt.tan(self.min)

        #dfc = np.tanh(abs(final_tan_theta)) * 0.5 + 0.5
        #x_new = x - self.lrn_rate * m_adj * dfc / (np.sqrt(v_adj) + self.eps)
        #self.g_prev = g
        #return x_new
