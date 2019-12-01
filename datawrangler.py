import numpy as np
import functools
import pandas as pd
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt

SAMPLE_IDX = 31128

class Normalizer(object):
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []
        self.padding = 5
        self.samples = []
    def apply(self,data):
        col_width = np.max([len(type(l).__name__) for l in self.layers]) + self.padding
        in_width = 20
        out_width = 20
        header = "|".join(["Module".center(col_width),"Input Shape".center(in_width),"Output Shape".center(out_width)])
        header_bar = "-"*len(header)
        print(header)
        print(header_bar)
        for layer in self.layers:
            data,sample_tup = layer.apply(data,col_width=col_width,in_width=in_width,out_width=out_width)
            self.samples.append(sample_tup)
        print(header_bar)
        return data

    def plot_samples(self):
        #fig,axes = plt.subplots(nrows=len(self.samples),figsize=(9,7*len(self.samples)))

        for idx,(smpl,mt_name) in enumerate(self.samples):
            plt.figure(figsize=(9,7))
            plt.plot(smpl.values)
            plt.title("Subject %d after Normalization Step %d (%s)" % (SAMPLE_IDX,idx,mt_name))
            plt.savefig("Step_%d.PNG" % idx)
            plt.close()


class Method(object):

    @classmethod
    def normalization_method(self, func):
        def inner(self,data,col_width,in_width,out_width,*args,**kwargs):
            input_shape = str(data.shape).center(in_width)
            method_name = type(self).__name__.ljust(col_width)
            data = func(self,data,*args,**kwargs)
            output_shape = str(data.shape).center(out_width)
            method_str = "|".join([method_name,input_shape,output_shape])
            print(method_str)
            sample = data.loc[SAMPLE_IDX,:]
            return data,(sample,method_name.replace(" ",""))
        return inner


class ZTransformNormalize(Method):
    def __init__(self,axis):
        self.axis = axis
        super().__init__()

    @Method.normalization_method
    def apply(self,data):
        data = pd.DataFrame((data.values - data.values.mean(axis=self.axis,keepdims=True))/data.values.std(axis=self.axis,keepdims=True),columns=data.columns,index=data.index)
        return data

class MinMaxNormalize(Method):
    def __init__(self,axis):
        self.axis = axis

    @Method.normalization_method
    def apply(self,data):
        data_mins = np.min(data.values,axis=self.axis,keepdims=True)
        data_maxs = np.max(data.values,axis=self.axis,keepdims=True)
        data = pd.DataFrame((data.values - data_mins)/(data_maxs - data_mins),columns=data.columns,index=data.index)
        return data

class NaNReplacer(Method):
    def __init__(self,const_val=0):
        self.const_val = const_val
    @Method.normalization_method
    def apply(self,data):
        nans = np.isnan(data)
        data[nans] = self.const_val
        return data

class ConstValueDropper(Method):
    def __init__(self,axis):
        self.axis = axis

    @Method.normalization_method
    def apply(self,data):
        is_const_val = np.all(np.equal(data.values,np.min(data.values,axis=self.axis,keepdims=True)),axis=self.axis)
        data = data[~is_const_val]
        return data

class LongToWideFormat(Method):
    def __init__(self,index_col,data_col,timestamp_col):
        self.index_col = index_col
        self.data_col = data_col
        self.timestamp_col = timestamp_col

    @Method.normalization_method
    def apply(self,data):
        return data.pivot(index=self.index_col,values=self.data_col,columns=self.timestamp_col)

class TimeseriesResampler(Method):
    def __init__(self,args,**kwargs):
        self.args = args
        self.kwargs = kwargs

    @Method.normalization_method
    def apply(self,data):
        return data.resample(self.args,**self.kwargs).sum()

class StableSeasonalFilter(Method):
    def __init__(self,num_seasons):
        self.num_seasons = num_seasons

    @Method.normalization_method
    def apply(self,data):
        tmp = data.values
        len_season = tmp.shape[1]//self.num_seasons

        assert (tmp.shape[1] == len_season * self.num_seasons)

        conv_filt = np.full(len_season + 1, 1 / len_season)
        conv_filt[0] = 1 / (len_season * 2)
        conv_filt[-1] = 1 / (len_season * 2)
        conv_filt = np.repeat(conv_filt, tmp.shape[0]).reshape(-1, tmp.shape[0]).transpose()
        moving_avg = fftconvolve(tmp, conv_filt, "same", axes=-1)

        detrended = tmp - moving_avg

        mean_idcs = np.repeat(np.arange(len_season), self.num_seasons).reshape(-1, self.num_seasons) + (
                    np.arange(self.num_seasons) * len_season)

        mean_comps = np.expand_dims(detrended[:, mean_idcs].mean(axis=-1), axis=0)

        stab_seas_comp = np.repeat(mean_comps, self.num_seasons, axis=0).transpose(1, 0, 2).reshape(tmp.shape[0], -1)
        filtered = tmp - stab_seas_comp
        return pd.DataFrame(filtered,index=data.index,columns=data.columns)

if __name__ == "__main__":

    data = pd.read_csv("C:\\Users\\96ahi\\Documents\\DataManipulate\\PAXRAW_D\\PAXRAW_D.csv")
    # Only use week 2 data
    data = data[data["Week"] == 2].reset_index(drop=True)
    # Get datetimes
    data["Datetime"] = pd.to_datetime(data["Datetime"],format="%d%b%y:%H:%M:%S")
    # Add a timedelta index (measure time the same for each patient, regardless of when they started)
    data["delta_t"] = data.groupby("Subid").apply(lambda grp: grp.Datetime - grp.Datetime.iloc[0]).reset_index(drop=True)


    normalizer = Normalizer([LongToWideFormat(index_col="Subid",data_col="VMU",timestamp_col="delta_t"),
                             TimeseriesResampler("30T",axis=1),
                             NaNReplacer(const_val=0),
                             ConstValueDropper(axis=1),
                             StableSeasonalFilter(num_seasons=7),
                             ZTransformNormalize(axis=-1)
                             ])
    data_df = normalizer.apply(data)
    print("done!")


