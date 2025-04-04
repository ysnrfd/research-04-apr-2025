"""
ENGINE
=====

Neural Network NN or (nn) engine, tensor calc neural network

Engine NN
====
"""

class nn():
    
    def nnp(self, x, y, z, n_layer):
        #---------------------

        x = x
        y = y
        z = z
        n_layer = n_layer

        #-----------------------

        if x == 0:
            x = 1
        else:
            pass

        if y == 0:
            y = 1
        else:
            pass

        if z == 0:
            z = 1
        else:
            pass

        if n_layer == 0:
            print('L_ERR, Set To 1')
            n_layer = 1
        else:
            pass

        #----------------------

        dim = x*y*z
        g_layer = dim*n_layer

        #----------------------

        return g_layer
    
