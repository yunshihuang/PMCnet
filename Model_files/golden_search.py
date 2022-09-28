import numpy as np
from Model_files.functions import *

def golden_search(crit,reg_inf,reg_sup,tol,logger):
    
    # Minimisation de la fonction crit(10^Xip) par l'algorithme du nombre d'or    
    reg_list = [reg_inf,reg_sup]
    Loss_list = []


    tau     = 0.38197
    maxiter = -2.078*np.log(tol)
    
    myprint('====================A new training starts!=============',logger)

    myprint('maxiter for goldensearch is {}'.format(round(maxiter)),logger)

    reg_1 = (1-tau)*reg_inf + tau*reg_sup
    reg_2 = tau*reg_inf + (1-tau)*reg_sup
    
    myprint('the reg is {}'.format(reg_1),logger)
    J_1 = crit(reg_1)
    myprint('the reg is {}, the loss is {}'.format(reg_1,J_1),logger)
    myprint('the reg is {}'.format(reg_2),logger)
    J_2 = crit(reg_2)
    myprint('the reg is {}, the loss is {}'.format(reg_2,J_2),logger)
    Loss_list = [J_1,J_2]

    iteration = 1
    
    while iteration<maxiter:
        myprint('this is iteration {}'.format(iteration),logger)
        if J_1>J_2:
            reg_inf = reg_1
            reg_1   = reg_2
            J_1 = J_2
            reg_2 = tau * reg_inf + (1-tau)*reg_sup
            reg_list.append(reg_2)
            myprint('the reg is {}'.format(reg_2),logger)
            J_2 = crit(reg_2)
            Loss_list.append(J_2)
            myprint('the reg is {}, the loss is {}'.format(reg_2,J_2),logger)
        else:
            reg_sup = reg_2
            reg_2   = reg_1
            J_2 = J_1
            reg_1  = (1-tau)*reg_inf + tau*reg_sup
            reg_list.append(reg_1) 
            myprint('the reg is {}'.format(reg_1),logger)
            J_1 = crit(reg_1)
            Loss_list.append(J_1)
            myprint('the reg is {}, the loss is {}'.format(reg_1,J_1),logger)

        iteration = iteration+1
        
    if J_1>=J_2:
        reg = reg_2
        reg_list.append(reg_2)
        Loss_list.append(J_2)
    else:
        reg = reg_1
        reg_list.append(reg_1)
        Loss_list.append(J_1)


    return reg,reg_list,Loss_list


