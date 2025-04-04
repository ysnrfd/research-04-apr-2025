class extra_engine():
    def ext_engine(self, n_layer, fwd_layer, bcw_layer):
        n_layer = n_layer
        fwd_layer = fwd_layer
        bcw_layer = bcw_layer
        #------------------------
        mult = n_layer**fwd_layer
        sum_stg1 = mult+bcw_layer
        sum_stg2 = fwd_layer*bcw_layer
        fine_sum = sum_stg1+sum_stg2
        #----------------------------
        return fine_sum
    