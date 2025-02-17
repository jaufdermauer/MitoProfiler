import numpy as np

import tkinter as tk
from tkinter import ttk

import copy





class XY_plot:
    def __init__ (self, x_arg, y_arg):
        self.x = x_arg
        self.y = y_arg

class fcs_channel:
    
    def __init__ (self, name_arg, fluct_arr_arg, auto_corr_arr_arg, short_name_arg):
        self.name = name_arg
        self.fluct_arr = fluct_arr_arg
        self.auto_corr_arr = auto_corr_arr_arg
        self.short_name = short_name_arg

        

        cr_list = []


        ys = 0
        counter = 1

        for i in range(len(self.fluct_arr.x)):
        
            ys += self.fluct_arr.y[i]

            if self.fluct_arr.x[i] >= counter:
                cr_list.append(ys)
                ys = 0
                counter +=1

        self.count_rate = np.mean(cr_list)/1000




class fcs_cross:
    
    def __init__(self, name_arg,  cross_corr_arr_arg, short_name_arg):
        self.name = name_arg
        self.cross_corr_arr = cross_corr_arr_arg
        self.short_name = short_name_arg


    

class Dataset_fcs:
    
    def __init__ (self,channels_number_arg, cross_number_arg, channels_list_arg, cross_list_arg):
        self.channels_number = channels_number_arg
        self.cross_number = cross_number_arg
        self.channels_list = channels_list_arg
        self.cross_list = cross_list_arg

class Full_dataset_fcs:
    
    def __init__ (self, repetitions_arg, dataset_list_arg):



        self.position = ''


        self.repetitions = repetitions_arg
        self.datasets_list = dataset_list_arg
        self.threshold_list = [None] * self.datasets_list[0].channels_number
        
        self.binning = 1
        self.peaks = {}
        self.peak_prominences = {}
        self.peak_widths = {}
        self.gp_fitting = [None] * repetitions_arg
        self.diff_fitting = {}
        self.N = {}
        self.cpm = {}
        self.diff_coeffs = {}

        for i in range(self.datasets_list[0].channels_number + self.datasets_list[0].cross_number):
            for j in range(repetitions_arg):
                self.diff_fitting[j, i] = None
                self.diff_coeffs[j, i] = None



        for i in range(self.datasets_list[0].channels_number):
            for j in range(repetitions_arg):
                self.N[j, i] = None
                self.cpm[j, i] = None

        for i in range(self.datasets_list[0].channels_number):
            for j in range(repetitions_arg):
                self.peaks[j, i] = None




        
    
        
#---------------------------------------------------  
        
#---------------------------------------------------

def Fill_datasets_fcs( list_file):

    #print ("Begin")

    current_repetition = 0

    i=0

    channels_fluct_list = []
    channels_cross_list = []
    dataset_list=[]
    full_dataset_list=[]

    array_size_min = -1

    while i < len(list_file):



        if list_file[i].__contains__("CarrierRows"):

            str1 , str2 = list_file[i].split(' = ')
            CarrierRows = int(str2)

            str1 , str2 = list_file[i+1].split(' = ')
            CarrierColumns = int(str2)

            positions = CarrierRows*CarrierColumns


            break

        i +=1



    position = 0
    i = 0

    while i < len(list_file):

        
        if list_file[i].__contains__("Repetition"):

            

            str1 , str2 = list_file[i].split(' = ')
            repetition = int(str2)

            #print ("Repetition ", repetition)

            if repetition > current_repetition and repetition != -1:

                flag = 0


                dataset_list.append(Dataset_fcs(len(channels_fluct_list), len(channels_cross_list), channels_fluct_list, channels_cross_list))
                current_repetition = repetition
                channels_fluct_list = []
                channels_cross_list = []

            if repetition == current_repetition and repetition != -1:

                str1 , position = list_file[i-2].split(' = ')
                position = int(position)

                str1 , long_name = list_file[i+1].split(' = ')


                

                
                


                if list_file[i+1].__contains__("versus"):

                    str1, str2 = long_name.split(" versus ")
                    str3, str4 = str1.split("Meta")

                    if len(str2.split("Meta")) == 2:
                        str5, str6 = str2.split("Meta")
                    elif len(str2.split("detector Ch")) == 2:
                        str5, str6 = str2.split("detector Ch")
                    
                    short_name = "channel " + str4 + " vs " + str(int(str6))

                    str1 , str2 = list_file[i+7].split(' = ')
                    corr_array_size = int(str2)
                    array_corr = list_file[i+9:i+9+corr_array_size]

                    x = []
                    y = []

                    for j in range(len(array_corr)):
                        str1, str2 = array_corr[j].split()
                        x.append(float(str1))
                        y.append(float(str2))

                    array_corr = XY_plot(x,y)

                    channel = fcs_cross(long_name,  array_corr, short_name)

                    channels_cross_list.append(channel)

                    i = i+9+corr_array_size

                    #print (long_name, corr_array_size)



                else:
                    if len(long_name.split("Meta")) == 2:
                        str1, str2 = long_name.split("Meta")
                    elif len(long_name.split("detector Ch")) == 2:
                        str5, str6 = long_name.split("detector Ch")

                    short_name = "channel " + str(int(str2))
                    

                    str1 , str2 = list_file[i+5].split(' = ')
                    array_size = int(str2)

                    if array_size < array_size_min or array_size_min == -1:
                        array_size_min = array_size

                    array_fluct =list_file[i+7:i+7+array_size]

                    
                   

                    str1 , str2 = list_file[i+7+array_size].split(' = ')
                    corr_array_size = int(str2)
                    

                    #print (long_name, array_size, corr_array_size)

                    array_corr = list_file[i+7+array_size+2:i+7+array_size+2+corr_array_size]

                    x = []
                    y = []

                    for j in range(len(array_fluct)):
                        str1, str2 = array_fluct[j].split()
                        x.append(float(str1))
                        y.append(float(str2))

                    array_fluct = XY_plot(x,y)

                    x = []
                    y = []

                    for j in range(len(array_corr)):
                        str1, str2 = array_corr[j].split()
                        x.append(float(str1))
                        y.append(float(str2))

                    array_corr = XY_plot(x,y)

                    channel = fcs_channel(long_name, array_fluct, array_corr, short_name)

                    channels_fluct_list.append(channel)

                    i = i+7+array_size+2+corr_array_size

                



                i+=1

            if repetition == -1 and flag != 1:

                flag = 1
                dataset_list.append(Dataset_fcs(len(channels_fluct_list), len(channels_cross_list), channels_fluct_list, channels_cross_list))

                repetitions = current_repetition+1

                current_repetition = 0




                for item1 in dataset_list:
                    for item2 in item1.channels_list:
                        del item2.fluct_arr.x[array_size_min-1 : -1]
                        del item2.fluct_arr.y[array_size_min-1 : -1]

                full_dataset = Full_dataset_fcs(repetitions, dataset_list)


                full_dataset.position = str(chr((position)//6 + 65)) + "_" + str((position)%6 + 1)

                full_dataset_list.append(full_dataset)

                print("position imported: ", full_dataset.position)



                channels_fluct_list = []
                channels_cross_list = []

                dataset_list = []
                

                if position == positions -1:           

                    
                    
                    break


                i+=1

                continue

        i+=1





    return full_dataset_list