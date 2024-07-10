import tifffile
import czifile
import matplotlib.pyplot as plt
import numpy as np
import Correlation as corr_py
import lmfit
import pandas as pd
from datetime import datetime
from fluct_prof import Analyse_sFCS_data_MLE_BIC as data_an


class File_sFCS:
    def __init__(self,lsm_file_name):
        print(lsm_file_name)
        self.lsm_file_name = lsm_file_name
        #read 1 line scanning FCS files
        #read CZI
        if lsm_file_name.endswith("czi"):
            image = czifile.imread(self.lsm_file_name)
            #image[0,t,c,0,0,y,0]
            reshaped_image = image[0, :, :, 0, 0, :, 0]
            reshaped_image = reshaped_image.transpose(1, 0, 2)
            self.array = reshaped_image

        #read TIF
        elif lsm_file_name.endswith("tif"):
            image = tifffile.imread(self.lsm_file_name)
            if len(image.shape) == 3:
                self.array =  tifffile.imread(self.lsm_file_name)
            elif len(image.shape) == 4:
                self.array =  image.reshape((image.shape[1], image.shape[0], image.shape[3]))
                print("tif reshaped")

        #read LSM
        else:
            self.array = tifffile.imread(self.lsm_file_name, key = 0)
            print(self.array.shape)

    def isolate_channel(self,channel_no):
        print("array shape: ", self.array.shape)
        if len(self.array.shape) == 2:
            return self.array
        else:
            return self.array[channel_no-1]
    
    def spatial_bin(self,channel_no,bin_size):#resulting array has intensities by rows
        channel = self.isolate_channel(channel_no)
        print("channel shape: ", channel.shape)
        num_bins = channel.shape[1] // bin_size
        reshaped_channel = channel[:, :num_bins * bin_size].reshape(channel.shape[0], num_bins, bin_size)
        binned_array = np.transpose(reshaped_channel.sum(axis=2))

        print("binned array shape: ", binned_array.shape)
        return binned_array
    
    def slice_in_time(self, channel_no, bin_size, n_slices):
        binned_array = self.spatial_bin(channel_no,bin_size)
        print("now slice in time")
        # Calculate the size of each slice
        slice_size = binned_array.shape[1] // n_slices

        # Reshape the binned_array for slicing
        sliced_array = binned_array[:, :slice_size * n_slices].reshape(-1, slice_size)

        # Handle any remaining elements that don't fit into slices
        remaining_elements_count = binned_array.shape[1] % n_slices
        if remaining_elements_count != 0:
            remaining_elements = binned_array[:, slice_size * n_slices:]
            remaining_elements_reshaped = np.vstack(np.array_split(remaining_elements, remaining_elements.shape[1], axis=1))
            sliced_array = np.vstack((sliced_array, remaining_elements_reshaped))

        print("sliced array shape: ", sliced_array.shape)
        return sliced_array
    
    def intensity_carpet_plot(self,channel_no, bin_size = 1, n_slices = 1):
        binned_data = self.slice_in_time(channel_no, bin_size, n_slices)
        return binned_data
        
    
    def plot_signal(self, channel_no, pixel, bin_size = 1, n_slices = 1): #for binned data, but works for unbinned too!
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        ##Could plot time (partly from from Falk's data):
        scanning_frequency = 2000 #2069 #Hz(global parameter) #from Falk
        line_dur = ((1.0/scanning_frequency)*1000.0) #from Falk
        x = [*range(0,len(binned_array[0]))]
        x = np.array(x)
        time_ms = x * line_dur #gives time in ms
        y = binned_array[pixel]
        plt.figure(figsize=(15,4))
        #plt.plot(x,y)
        plt.plot(time_ms,y)
        plt.ylabel ('Intensity')
        plt.xlabel ('Time (ms)')
        plt.title ('Intensity trace for row no. {}'.format(str(pixel)))
        plt.tight_layout()
        plt.show()    
        return time_ms,y 

    def single_trace(self, channel_no, timestep, row, bin_size = 1, n_slices=1):
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        y = binned_array[row]
        x = np.linspace(0, (len(y)-1)*timestep, len(y))
        plt.plot (x, y)
        plt.xlabel('Time (ms)')
        plt.ylabel('Intensity (kHz)')
        plt.tight_layout()
        plt.show()
        return x,y

    
    def single_autoc_plot(self, channel_no, timestep, row, bin_size = 1, n_slices=1):
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        y = binned_array[row]
        time, scorr = corr_py.correlate_full (timestep, y, y)
        plt.xscale("log")
        plt.plot (time, scorr)
        plt.xlabel('Delay Time (ms)')
        plt.ylabel('G (tau)')
        plt.tight_layout()
        plt.show()
        return time,scorr

    def autoc_carpet_plot(self, channel_no, timestep, bin_size = 1, n_slices = 1,plot_title =''):
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        autocorrelation_by_rows = []
        for i in range(len(binned_array)):
            y = binned_array[i]
            time, scorr = corr_py.correlate_full (timestep, y, y)
            autocorrelation_by_rows.append(scorr)
        fig, ax = plt.subplots(figsize=(100,10))
        im = ax.imshow(autocorrelation_by_rows,origin="lower",cmap='bwr')
        #cbar = ax.figure.colorbar(im, ax=ax,shrink=0.5,location='right', pad =0.003)
        ax.set_title(plot_title)
        plt.show()
        #return autocorrelation_by_rows
    
    def get_fitting_params(self, channel_no, timestep, bin_size, n_slices, 
                           input_params, method='least_squares', export=False):
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        self.params_per_row =[]
        list_of_keys = ["Row_no"]
        for i in input_params.keys():
            list_of_keys.append(i)
        list_of_keys.append('Chi_Sqr')
        self.params_per_row.append(list_of_keys)
        for row in range(len(binned_array)):
            y = binned_array[row]
            time, scorr = corr_py.correlate_full(timestep,y,y)
            o1 = lmfit.minimize(resid,input_params,args=(time,scorr),method=method)
            params_in_row = []
            params_in_row.append(row+1)
            for param in input_params.keys():
                if param == 'txy':
                    params_in_row.append(np.float64(o1.params[param].value)*1000)
                else:
                    params_in_row.append(np.float64(o1.params[param].value))
            params_in_row.append(np.float64(o1.chisqr))
            self.params_per_row.append(params_in_row)
        if export == True:
            export_file_name = "{date}_Ch{channel_no}_{bin_size}bins_{n_slices}slices.csv".format(
                channel_no=channel_no, bin_size=bin_size, n_slices=n_slices, date=datetime.date(datetime.now()))
            self.params_to_csv(export_file_name)
        list_of_keys = self.params_per_row[0]
        param_dict = {}
        for i in list_of_keys:
            param_dict[i]=[]
        for row in range(1,len(self.params_per_row)):
            counter=0
            for i in list_of_keys:
                param_dict[i].append(self.params_per_row[row][counter])
                counter+=1
        self.params_df = pd.DataFrame(param_dict,index=param_dict['Row_no'])
        return self.params_df

    def params_to_csv(self, export_file_name):
        with open (export_file_name,"w") as f:
            for i in self.params_per_row:    
                f.write(','.join([str(k) for k in i]))
                f.write('\n')

    def fitting_figure(self, cutoff_start = 0, cutoff_end=500):
        try:
            cutoff = np.array ([cutoff_start, cutoff_end]) # Upper and lower bounds for transit times. Do not consider too small or too big transit times
            data = self.params_df[self.params_df['txy']< cutoff[1]]
            data = self.params_df[self.params_df['txy']> cutoff[0]]
            results, fig = data_an.model_selection_RL (data['txy'], initial_guess = [4,1], plot = 'on')
            return results
        except:
            print("Error, call File_sFCS.get_fitting_params(....) method before then try again")





class File_2fsFCS:
    def __init__(self,lsm_file_name):
        print(lsm_file_name)
        self.lsm_file_name = lsm_file_name
        #read 2 line scanning FCS files

        print("File_2fsFCS")
        #read CZI 2 line scanning file
        if lsm_file_name.endswith("czi"):
            image = czifile.imread(self.lsm_file_name)
            print(image.shape)
            if image.shape[2] == 4:	#2fsFCCS
                reshaped_image = np.empty((2, image.shape[1], 2, image.shape[5]), dtype = float)
                for c in range(2):
                    for y in range(image.shape[5]):
                        for t in range(image.shape[1]):
                            reshaped_image[c,t,0,y] = image[0,t,2*c+1,0,0,y,0] #first focus line
                            reshaped_image[c,t,1,y] = image[0,t,2*c+1,0,-1,y,0] #second focus line
            else:
                reshaped_image = np.empty((image.shape[2], image.shape[1], 2, image.shape[5]), dtype = float)
                for c in range(image.shape[2]):
                    for y in range(image.shape[5]):
                        for t in range(image.shape[1]):
                            reshaped_image[c,t,0,y] = image[0,t,c,0,0,y,0] #first focus line
                            reshaped_image[c,t,1,y] = image[0,t,c,0,-1,y,0] #second focus line
            self.array = reshaped_image


        #read TIF
        elif lsm_file_name.endswith("tif"):
            image = tifffile.imread(self.lsm_file_name)
            print(image.shape)
            self.array =  tifffile.imread(self.lsm_file_name)

        #read LSM
        else:
            self.array = tifffile.imread(self.lsm_file_name, key = 0)
            print(self.array.shape)

    def isolate_channel(self,channel_no):
        print("isolate channel")
        print(self.array.shape)
        if len(self.array.shape) == 2 or (self.array.shape[0] > 2 and len(self.array.shape) == 3):
            return self.array
        else:
            return self.array[channel_no-1]
    
    def spatial_bin(self,channel_no, line_no, bin_size):#resulting array has intensities by rows
        channel = self.isolate_channel(channel_no)
        print(channel.shape)
        print("spatial bin")
        i = 0
        len_array = channel.shape[0]
        print(len_array)
        #len_array = len(channel[:,0,0])
        binned_array = np.zeros((len_array), dtype = float)
        #binned_array2 = np.zeros((len(channel[:,0,0])), dtype = float)
        print(binned_array.shape)
        while i < channel.shape[-1]-bin_size+1:
            j = 0
            addition_array = np.zeros((len_array), dtype = float)
            addition_array2 = np.zeros((len_array), dtype = float)
            while j < bin_size:
                if len(channel.shape) == 3: #2 colors
                    addition_array += channel[:,line_no, i]
                elif len(channel.shape) == 2: #1 color
                    addition_array += channel[:,line_no, i]
                #addition_array2 += channel[:,1, i]
                j += 1
                i += 1
            binned_array = np.row_stack((binned_array,addition_array))
            #binned_array2 = np.row_stack((binned_array2,addition_array2))
        binned_array = np.delete(binned_array,(0),axis=0)
        #binned_array2 = np.delete(binned_array2,(0),axis=0)
        #binned_array = np.array([binned_array, binned_array2])
        print(binned_array.shape)

        return binned_array
    
    def slice_in_time(self, channel_no, line_no, bin_size, n_slices):
        binned_array = self.spatial_bin(channel_no,line_no,bin_size)
        sliced_array = np.zeros(int(len(binned_array[0])/n_slices))
        for i in range(len(binned_array[:,0])):
            arr = np.array_split(binned_array[i],n_slices)
            arr = np.vstack(arr)
            sliced_array = np.row_stack((sliced_array,arr))
        sliced_array = np.delete(sliced_array,(0),axis=0)
        return sliced_array
    
    def intensity_carpet_plot(self,channel_no, line_no, bin_size = 1, n_slices = 1):
        binned_data = self.slice_in_time(channel_no, line_no, bin_size, n_slices)
        return binned_data
        
    
    def plot_signal(self, channel_no, line_no, pixel, bin_size = 1, n_slices = 1): #for binned data, but works for unbinned too!
        binned_array = self.slice_in_time(channel_no, line_no, bin_size, n_slices)
        ##Could plot time (partly from from Falk's data):
        scanning_frequency = 2000 #2069 #Hz(global parameter) #from Falk
        line_dur = ((1.0/scanning_frequency)*1000.0) #from Falk
        x = [*range(0,len(binned_array[0]))]
        x = np.array(x)
        time_ms = x * line_dur #gives time in ms
        y = binned_array[pixel]
        plt.figure(figsize=(15,4))
        #plt.plot(x,y)
        plt.plot(time_ms,y)
        plt.ylabel ('Intensity')
        plt.xlabel ('Time (ms)')
        plt.title ('Intensity trace for row no. {}'.format(str(pixel)))
        plt.tight_layout()
        plt.show()    
        return time_ms,y 

    def single_trace(self, channel_no, timestep, row, bin_size = 1, n_slices=1):
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        y = binned_array[row]
        x = np.linspace(0, (len(y)-1)*timestep, len(y))
        plt.plot (x, y)
        plt.xlabel('Time (ms)')
        plt.ylabel('Intensity (kHz)')
        plt.tight_layout()
        plt.show()
        return x,y

    
    def single_autoc_plot(self, channel_no, timestep, row, bin_size = 1, n_slices=1):
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        y = binned_array[row]
        time, scorr = corr_py.correlate_full (timestep, y, y)
        plt.xscale("log")
        plt.plot (time, scorr)
        plt.xlabel('Delay Time (ms)')
        plt.ylabel('G (tau)')
        plt.tight_layout()
        plt.show()
        return time,scorr

    def autoc_carpet_plot(self, channel_no, timestep, bin_size = 1, n_slices = 1,plot_title =''):
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        autocorrelation_by_rows = []
        for i in range(len(binned_array)):
            y = binned_array[i]
            time, scorr = corr_py.correlate_full (timestep, y, y)
            autocorrelation_by_rows.append(scorr)
        fig, ax = plt.subplots(figsize=(100,10))
        im = ax.imshow(autocorrelation_by_rows,origin="lower",cmap='bwr')
        #cbar = ax.figure.colorbar(im, ax=ax,shrink=0.5,location='right', pad =0.003)
        ax.set_title(plot_title)
        plt.show()
        #return autocorrelation_by_rows
    
    def get_fitting_params(self, channel_no, timestep, bin_size, n_slices, 
                           input_params, method='least_squares', export=False):
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        self.params_per_row =[]
        list_of_keys = ["Row_no"]
        for i in input_params.keys():
            list_of_keys.append(i)
        list_of_keys.append('Chi_Sqr')
        self.params_per_row.append(list_of_keys)
        for row in range(len(binned_array)):
            y = binned_array[row]
            time, scorr = corr_py.correlate_full(timestep,y,y)
            o1 = lmfit.minimize(resid,input_params,args=(time,scorr),method=method)
            params_in_row = []
            params_in_row.append(row+1)
            for param in input_params.keys():
                if param == 'txy':
                    params_in_row.append(np.float64(o1.params[param].value)*1000)
                else:
                    params_in_row.append(np.float64(o1.params[param].value))
            params_in_row.append(np.float64(o1.chisqr))
            self.params_per_row.append(params_in_row)
        if export == True:
            export_file_name = "{date}_Ch{channel_no}_{bin_size}bins_{n_slices}slices.csv".format(
                channel_no=channel_no, bin_size=bin_size, n_slices=n_slices, date=datetime.date(datetime.now()))
            self.params_to_csv(export_file_name)
        list_of_keys = self.params_per_row[0]
        param_dict = {}
        for i in list_of_keys:
            param_dict[i]=[]
        for row in range(1,len(self.params_per_row)):
            counter=0
            for i in list_of_keys:
                param_dict[i].append(self.params_per_row[row][counter])
                counter+=1
        self.params_df = pd.DataFrame(param_dict,index=param_dict['Row_no'])
        return self.params_df

    def params_to_csv(self, export_file_name):
        with open (export_file_name,"w") as f:
            for i in self.params_per_row:    
                f.write(','.join([str(k) for k in i]))
                f.write('\n')

    def fitting_figure(self, cutoff_start = 0, cutoff_end=500):
        try:
            cutoff = np.array ([cutoff_start, cutoff_end]) # Upper and lower bounds for transit times. Do not consider too small or too big transit times
            data = self.params_df[self.params_df['txy']< cutoff[1]]
            data = self.params_df[self.params_df['txy']> cutoff[0]]
            results, fig = data_an.model_selection_RL (data['txy'], initial_guess = [4,1], plot = 'on')
            return results
        except:
            print("Error, call File_sFCS.get_fitting_params(....) method before then try again")





def Corr_curve_2d(tc, offset, GN0, A1, txy1, alpha1, B1, tauT1):
    txy1 = txy1
    tauT1 = tauT1
    G_Diff =  A1*(((1+((tc/txy1)**alpha1))**-1))    #autocorrelation in 2D
    G_T = 1 + (B1*np.exp(tc/(-tauT1)))
    return offset + GN0 * G_Diff * G_T

def resid (params, x, ydata ):
    param_list = []    
    for param in params.keys():
        param_list.append( np.float64(params[param].value))
        
    y_model = Corr_curve_2d(x, *param_list)
    return y_model - ydata

def params_lists_to_object(list_of_params, list_of_inits, list_of_vary, list_of_min, list_of_max):
    params = lmfit.Parameters()
    for i in range(0, len(list_of_params)):
        params.add(list_of_params[i], float(list_of_inits[i]), vary = int(list_of_vary[i]), 
                   min = float(list_of_min[i]), max = float(list_of_max[i]))
    return params

## 2fsFCCS correlation functions ##

#2fsFCCS CC curve between lines with old parameters
def CC_2FSFCCS_old(tc, offset, GN0, A1, txy1, alpha1, B1, tauT1, d, w0):
    txy1 = txy1
    tauT1 = tauT1
    G_Diff =  A1*(((1+((tc/txy1)**alpha1))**-1))    #autocorrelation in 2D
    G_d = np.exp(-d**2/(w0**2+w0**2*tc/txy1))
    return offset + GN0 * G_Diff * G_d

#new parameters:
# C = N / (pi**(3/2)*w0**3*S) concentration
# S: structural parameter
# w0 waist radius
# D: diffusion coefficient
# tau: time shift
# d: distance between 2 lines
# tau_D = w0**2 / (4*D) diffusion time
# tau_T: triplet time
# T: triplet state population

#sFCCS CC curve in 2 dimensions
def CC_FCCS_2d(tau, offset, D, tau_T, T, w0, S, C):
    G_auto = 1/(C*np.pi**(3/2)*w0**3*S)*(1+4*D*tau/w0**2)**(-1/2)*(1+4*D*tau/(S**2*w0**2))**(-1/2)
    G_T = 1 + T/(1-T) * np.exp(-tau/(tau_T))
    return offset + G_auto * G_T

#2fsFCCS CC curve between lines with new parameters
def CC_2fsFCCS_2d(tau, offset, C, S, D, w0, d):
    D = D * 1000000
    G_auto = 1/(C*np.pi*S*w0**2)*(1+4*D*tau/w0**2)**(-1/2)*(1+4*D*tau/(S**2*w0**2))**(-1/2)
    G_CC = np.exp(-d**2/(4*D*tau+w0**2))
    return offset + G_auto * G_CC