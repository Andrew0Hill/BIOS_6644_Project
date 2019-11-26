import numpy as np
import functools
import pandas as pd

class Normalizer(object):
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []
        self.padding = 5
    def apply(self,data):
        col_width = np.max([len(type(l).__name__) for l in self.layers]) + self.padding
        in_width = 20
        out_width = 20
        header = "|".join(["Module".center(col_width),"Input Shape".center(in_width),"Output Shape".center(out_width)])
        header_bar = "-"*len(header)
        print(header)
        print(header_bar)
        for layer in self.layers:
            data = layer.apply(data,col_width=col_width,in_width=in_width,out_width=out_width)
        print(header_bar)
        return data



class Method(object):

    @classmethod
    def test_dec(self,func):
        def inner(self,data,col_width,in_width,out_width,*args,**kwargs):
            input_shape = str(data.shape).center(in_width)
            method_name = type(self).__name__.ljust(col_width)
            data = func(self,data,*args,**kwargs)
            output_shape = str(data.shape).center(out_width)
            method_str = "|".join([method_name,input_shape,output_shape])
            print(method_str)
            return data
        return inner


class ZTransformNormalize(Method):
    def __init__(self,axis):
        self.axis = axis
        super().__init__()

    @Method.test_dec
    def apply(self,data):
        data[:] = (data - data.mean(axis=self.axis,keepdims=True))/data.std(axis=self.axis,keepdims=True)
        return data

class MinMaxNormalize(Method):
    def __init__(self,axis):
        self.axis = axis

    @Method.test_dec
    def apply(self,data):
        data_mins = np.min(data.values,axis=self.axis,keepdims=True)
        data_maxs = np.max(data.values,axis=self.axis,keepdims=True)
        data = pd.DataFrame((data.values - data_mins)/(data_maxs - data_mins),columns=data.columns,index=data.index)
        return data

class NaNReplacer(Method):
    def __init__(self,const_val=0):
        self.const_val = const_val
    @Method.test_dec
    def apply(self,data):
        nans = np.isnan(data)
        data[nans] = self.const_val
        return data

class ConstValueDropper(Method):
    def __init__(self,axis):
        self.axis = axis

    @Method.test_dec
    def apply(self,data):
        is_const_val = np.all(np.equal(data.values,np.min(data.values,axis=self.axis,keepdims=True)),axis=self.axis)
        data = data[~is_const_val]
        return data


if __name__ == "__main__":

    data = np.random.random((7400,300))
    # Simulate constant value rows
    data[np.arange(500,600),:] = 6
    # Simulate NaN measurements
    data[np.arange(550,700),np.random.choice(data.shape[1],150)] = np.nan

    data_df = pd.DataFrame(data)
    normalizer = Normalizer([NaNReplacer(const_val=0),
                             ConstValueDropper(axis=1),
                             #ZTransformNormalize(axis=-1)
                             MinMaxNormalize(axis=-1)])
    data_df = normalizer.apply(data_df)
    print("done!")


