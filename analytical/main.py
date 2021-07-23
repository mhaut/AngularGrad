import matplotlib.pyplot as plt
import math as mt
import numpy as np
from optimizers import *
import os

def calc_func1(x):
    if x <= 0: val = (x + 0.3) ** 2
    else:      val = (x - 0.2) ** 2 + 0.05
    return val


def calc_grad1(x):
    if x <= 0: val = 2*x + 0.6
    else:      val = 2*x - 0.4
    return val


def calc_func2(x):
    if x <= -0.9: val = -40 * x - 35.15
    else:         val = (x * x * x) + x * mt.sin(8 * x) + 0.85
    return val


def calc_grad2(x):
    if x <= -0.9:
        return -40
    else:
        return 3 * x * x  + 8 * x * mt.cos(8 * x) + mt.sin(8 * x)


def calc_func3(x):
    if x <= -0.5:   val = x**2
    elif x <= -0.4: val = 0.75 + x
    elif x <= 0.0:  val = -7 * x / 8
    elif x <= 0.4:  val = 7 * x / 8
    elif x <= 0.5:  val = 0.75 - x
    else:           val = x**2
    return val


def calc_grad3(x):
    if x <= -0.5:   val = 2 * x
    elif x <= -0.4: val = 1.0
    elif x <= 0.0:  val = -7/8
    elif x <= 0.4:  val = 7/8
    elif x <= 0.5:  val = -1.0
    else:           val = 2 * x
    return val



def solve_func(xvals):
    return [calc_func(xval) for xval in xvals]


# optimize with the specified solver
def solve(x0, solver):
    x = np.zeros(nb_iters)
    x[0] = x0
    for idx_iter in range(1, nb_iters):
        g = calc_grad(x[idx_iter - 1])
        x[idx_iter] = solver.update(x[idx_iter - 1], g)
    return x

# optimize with the specified solver
def solve_reg(x0, solver):
    x = np.zeros(nb_iters)
    y = np.zeros(nb_iters)
    x[0] = x0
    for idx_iter in range(1, nb_iters):
        g = calc_grad(x[idx_iter - 1])
        x[idx_iter] = solver.update(x[idx_iter - 1], g)
        y[idx_iter] = calc_func(x[idx_iter])
    return x, y








nb_iters = 300
lrn_rate = 0.1
beta1 = 0.95
beta2 = 0.999
eps = 0.00000001
if not os.path.isdir('figures'):
    os.mkdir('figures')

for idfunc, calc_func in enumerate([calc_func1, calc_func2, calc_func3]):
    # Adam & diffGrad
    x = {}
    xvals = np.arange(-1,1,0.05)
    x['adam'] = solve_func(xvals)

    # visualization
    #plt.rcParams['figure.dpi']= 300
    plt.rcParams['figure.figsize'] = [6.0, 4.0]
    plt.plot(xvals, x['adam'], label='func')
    #plt.legend()
    plt.xlabel("x")
    plt.ylabel("F"+str(idfunc+1)+"(x)")
    plt.grid()
    #plt.show()
    plt.savefig('figures/function_'+str(idfunc)+'.png', dpi=600, format='png', bbox_inches='tight')
    plt.clf()


for idfunc, calc_grad in enumerate([calc_grad1, calc_grad2, calc_grad3]):
    # Adam & diffGrad
    x = {}
    x0 = -1.0
    solver = SGDM(lrn_rate, beta1, eps)
    x['sgdm'] = solve(x0, solver)

    x0 = -1.0
    solver = Adam(lrn_rate, beta1, beta2, eps)
    x['adam'] = solve(x0, solver)
    
    x0 = -1.0
    solver = diffGrad(lrn_rate, beta1, beta2, eps)
    x['diffGrad'] = solve(x0, solver)
    #solver = AdaBelief(lrn_rate, beta1, beta2, eps)
    #x['AdaBelief'] = solve(x0, solver)

    x0 = -1.0
    solver = AdaBelief(lrn_rate, beta1, beta2, eps)
    x['AdaBelief'] = solve(x0, solver)
    
    x0 = -1.0
    solver = AngularGradCos(lrn_rate, beta1, beta2, eps)
    x['AngularGradCos'] = solve(x0, solver)
    
    x0 = -1.0
    solver = AngularGradTan(lrn_rate, beta1, beta2, eps)
    x['AngularGradTan'] = solve(x0, solver)
    
    # visualization
    #plt.rcParams['figure.dpi']= 300
    plt.rcParams['figure.figsize'] = [6.0, 4.0]
    plt.plot(np.arange(nb_iters) + 1, x['sgdm'], label='SGDM')
    plt.plot(np.arange(nb_iters) + 1, x['adam'], label='Adam')
    plt.plot(np.arange(nb_iters) + 1, x['diffGrad'], label='diffGrad')
    plt.plot(np.arange(nb_iters) + 1, x['AdaBelief'], label='AdaBelief')
    plt.plot(np.arange(nb_iters) + 1, x['AngularGradCos'], label='$AngularGrad^{Cos}$')
    plt.plot(np.arange(nb_iters) + 1, x['AngularGradTan'], label='$AngularGrad^{Tan}$')
    plt.xlabel("Iteration")
    plt.ylabel("Parameters Values")
    plt.legend(ncol=2)
    plt.grid()
    #plt.show()
    plt.savefig('figures/deriv_'+str(idfunc)+'.png', dpi=600, format='png', bbox_inches='tight')
    plt.clf()


for idfunc, (calc_grad,calc_func) in enumerate(zip([calc_grad1, calc_grad2, calc_grad3], [calc_func1, calc_func2, calc_func3])):
    x = {}
    y = {}
    
    x0 = -1.0
    solver = SGDM(lrn_rate, beta1, eps)
    x['sgdm'], y['sgdm'] = solve_reg(x0, solver)
    
    x0 = -1.0
    solver = Adam(lrn_rate, beta1, beta2, eps)
    x['adam'], y['adam'] = solve_reg(x0, solver)
    
    x0 = -1.0
    solver = diffGrad(lrn_rate, beta1, beta2, eps)
    x['diffGrad'], y['diffGrad'] = solve_reg(x0, solver)

    x0 = -1.0
    solver = AdaBelief(lrn_rate, beta1, beta2, eps)
    x['AdaBelief'], y['AdaBelief'] = solve_reg(x0, solver)
    
    x0 = -1.0
    solver = AngularGradCos(lrn_rate, beta1, beta2, eps)
    x['AngularGradCos'], y['AngularGradCos'] = solve_reg(x0, solver)
    
    x0 = -1.0
    solver = AngularGradTan(lrn_rate, beta1, beta2, eps)
    x['AngularGradTan'], y['AngularGradTan'] = solve_reg(x0, solver)
    # visualization
    #plt.rcParams['figure.dpi']= 300
    plt.rcParams['figure.figsize'] = [6.0, 4.0]
    plt.plot(np.arange(nb_iters) + 1, y['sgdm'], label='SGDM')
    plt.plot(np.arange(nb_iters) + 1, y['adam'], label='Adam')
    plt.plot(np.arange(nb_iters) + 1, y['diffGrad'], label='diffGrad')
    plt.plot(np.arange(nb_iters) + 1, y['AdaBelief'], label='AdaBelief')
    plt.plot(np.arange(nb_iters) + 1, y['AngularGradCos'], label='$AngularGrad^{Cos}$')
    plt.plot(np.arange(nb_iters) + 1, y['AngularGradTan'], label='$AngularGrad^{Tan}$')
    plt.xlabel("Iteration")
    plt.ylabel("Regression Loss")
    #plt.legend()
    plt.legend(ncol=2)
    plt.grid()
    #plt.show()
    plt.savefig('figures/regression_'+str(idfunc)+'.png', dpi=600, format='png', bbox_inches='tight')
    plt.clf()

