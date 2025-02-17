#--------------------------
#Importing general modules
#--------------------------
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
import matplotlib.pyplot as plt

from scipy.stats import kde

import csv

import lmfit

import time


from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)


# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib import cm as mplcm

from ttkwidgets import CheckboxTreeview



import codecs

import os

from datetime import datetime

from scipy import stats

import copy

import numpy as np

from scipy.signal import find_peaks

from scipy.optimize import curve_fit
import random

import seaborn as sns


#--------------------------
#End of importing general modules
#--------------------------


#--------------------------
#Importing own modules
#--------------------------

from fluct_prof import fcs_importer

from fluct_prof import Correlation as corr_py

from fluct_prof import Functions as fun

from fluct_prof import Data_container as data_cont

from fluct_prof import Data_tree as d_tree

#--------------------------
#End of importing own modules
#--------------------------

def Norm():
	print("This is a temporary function")



class Dot_Plot_Window:

	def Save_plot_data(self):

		for key in self.thisdict_axis_1.keys():


			filename = data_cont.initialdirectory + "\\" +  key + "_Dot_Plot.txt"

			open_file = open(filename, 'w')


			open_file.write(str(key) + "\n")
			open_file.write(str(self.string_x) + "\t" + str(self.string_y) + "\n")

			for i in range(len(self.thisdict_axis_1[key])):
				open_file.write(str(self.thisdict_axis_1[key][i]) + "\t" + str(self.thisdict_axis_2[key][i]) + "\n")

			open_file.close()



	def Choose_dataset(self, event):




		index = self.tree.selection()

		num1, num = index[0].split('I')
		

		num = int(num, 16)

		

		sum1 = num 
		file = 0
		rep = 0
		for i in range (len(data_cont.data_list_raw)):
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (data_cont.repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1


		


		output_file_name = data_cont.tree_list_name[file1-1][:-4]




		file1 = file1-1
		rep1 = rep1-1




		output_file_name = data_cont.tree_list_name[file1-1][:-4]




		data_cont.file_index = file1-1
		data_cont.rep_index = rep1-1

		self.axis_choice = []


		

		if data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number > 1:
			for i in range (data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number):
				
				str1 = data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_list[i].short_name
				self.axis_choice.append(str1)
				self.axis_choice.append("prom_" + str1)
				self.axis_choice.append("width_" + str1)


		if data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number > 1:
			for i in range (data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number):
				
				str1 = "Diff_" + data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_list[i].short_name
				self.axis_choice.append(str1)

		if data_cont.data_list_raw[data_cont.file_index].datasets_list[0].cross_number > 1:
			for i in range (data_cont.data_list_raw[data_cont.file_index].datasets_list[0].cross_number):
				
				str1 = "Diff_" + data_cont.data_list_raw[data_cont.file_index].datasets_list[0].cross_list[i].short_name
				self.axis_choice.append(str1)

		self.axis_choice.append("GP")


		self.Axis_y_label__choice.config(values = self.axis_choice)
		self.Axis_x_label__choice.config(values = self.axis_choice)




		




	def Plot_dataset(self):





		self.dot_plot.cla()
		self.dens_plot.cla()
		self.colorbar.cla()



		list1 = self.tree.get_checked()

		



		self.thisdict_axis_1 = {}
		self.thisdict_axis_2 = {}

		for index in list1:

			num1, num = index.split('I')
			

			num = int(num, 16)

			

			sum1 = num 
			file = 0
			rep = 0
			for i in range (len(data_cont.data_list_raw)):
				rep = 0
				sum1-=1
				file+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep

				
				for j in range (data_cont.repetitions_list[i]):
					sum1-=1
					rep+=1
					if sum1 == 0:
						file1 = file
						rep1 = rep



			if rep1 == 0:
				rep1+=1


			
			

			output_file_name = data_cont.tree_list_name[file1-1][:-4]




			file1 = file1-1
			rep1 = rep1-1


			




		

			self.string_x = self.Axis_x_label__choice.get()
			self.string_y = self.Axis_y_label__choice.get()


			if self.string_x.__contains__("Diff") == True:

				str1, str2 = self.string_x.split("_")

				if data_cont.data_list_raw[file1].datasets_list[rep1].channels_number > 1:
					for i in range (data_cont.data_list_raw[file1].datasets_list[rep1].channels_number):
						
						if str2 == data_cont.data_list_raw[file1].datasets_list[rep1].channels_list[i].short_name:
							channel_number = i

				if data_cont.data_list_raw[file1].datasets_list[rep1].cross_number > 1:
					for i in range (data_cont.data_list_raw[file1].datasets_list[rep1].cross_number):
						
						if str2 == data_cont.data_list_raw[file1].datasets_list[rep1].cross_list[i].short_name:
							channel_number = i + data_cont.data_list_raw[file1].datasets_list[rep1].channels_number



				if output_file_name in self.thisdict_axis_1.keys():
					self.thisdict_axis_1[output_file_name].append(data_cont.data_list_raw[file1].diff_fitting[rep1, channel_number]["txy"])
				else:
					self.thisdict_axis_1[output_file_name] = []
					self.thisdict_axis_1[output_file_name].append(data_cont.data_list_raw[file1].diff_fitting[rep1, channel_number]["txy"])


			if self.string_y.__contains__("Diff") == True:

				str1, str2 = self.string_y.split("_")

				if data_cont.data_list_raw[file1].datasets_list[rep1].channels_number > 1:
					for i in range (data_cont.data_list_raw[file1].datasets_list[rep1].channels_number):
						
						if str2 == data_cont.data_list_raw[file1].datasets_list[rep1].channels_list[i].short_name:
							channel_number = i



				if data_cont.data_list_raw[file1].datasets_list[rep1].cross_number > 1:
					for i in range (data_cont.data_list_raw[file1].datasets_list[rep1].cross_number):
						
						if str2 == data_cont.data_list_raw[file1].datasets_list[rep1].cross_list[i].short_name:
							channel_number = i + data_cont.data_list_raw[file1].datasets_list[rep1].channels_number




				if output_file_name in self.thisdict_axis_2.keys():
					self.thisdict_axis_2[output_file_name].append(data_cont.data_list_raw[file1].diff_fitting[rep1, channel_number]["txy"])
				else:
					self.thisdict_axis_2[output_file_name] = []
					self.thisdict_axis_2[output_file_name].append(data_cont.data_list_raw[file1].diff_fitting[rep1, channel_number]["txy"])



			if self.string_x.__contains__("GP") == True:


				if output_file_name in self.thisdict_axis_1.keys():
					self.thisdict_axis_1[output_file_name].append(data_cont.data_list_raw[file1].gp_fitting[rep1]["Mean"])
				else:
					self.thisdict_axis_1[output_file_name] = []
					self.thisdict_axis_1[output_file_name].append(data_cont.data_list_raw[file1].gp_fitting[rep1]["Mean"])

			if self.string_y.__contains__("GP") == True:


				if output_file_name in self.thisdict_axis_2.keys():
					self.thisdict_axis_2[output_file_name].append(data_cont.data_list_raw[file1].gp_fitting[rep1]["Mean"])
				else:
					self.thisdict_axis_2[output_file_name] = []
					self.thisdict_axis_2[output_file_name].append(data_cont.data_list_raw[file1].gp_fitting[rep1]["Mean"])


			if self.string_x.__contains__("GP") == False and self.string_x.__contains__("Diff") == False:


				if self.string_x.__contains__("prom") == False and self.string_x.__contains__("width") == False:
					str1, str2 = self.string_x.split(" ")
					channel_number = int(str2) - 1

				

					if output_file_name in self.thisdict_axis_1.keys():
						self.thisdict_axis_1[output_file_name] = self.thisdict_axis_1[output_file_name] + data_cont.data_list_raw[file1].peaks[rep1, channel_number]
					else:
						self.thisdict_axis_1[output_file_name] = []
						self.thisdict_axis_1[output_file_name] = self.thisdict_axis_1[output_file_name] + data_cont.data_list_raw[file1].peaks[rep1, channel_number]

				if self.string_x.__contains__("prom") == True:
					str1, str2 = self.string_x.split(" ")
					channel_number = int(str2) - 1

					print (channel_number)

					if output_file_name in self.thisdict_axis_1.keys():
						self.thisdict_axis_1[output_file_name] = self.thisdict_axis_1[output_file_name] + data_cont.data_list_raw[file1].peak_prominences[rep1, channel_number]
					else:
						self.thisdict_axis_1[output_file_name] = []
						self.thisdict_axis_1[output_file_name] = self.thisdict_axis_1[output_file_name] + data_cont.data_list_raw[file1].peak_prominences[rep1, channel_number]

				if self.string_x.__contains__("width") == True:
					str1, str2 = self.string_x.split(" ")
					channel_number = int(str2) - 1

				

					if output_file_name in self.thisdict_axis_1.keys():
						self.thisdict_axis_1[output_file_name] = self.thisdict_axis_1[output_file_name] + data_cont.data_list_raw[file1].peak_widths[rep1, channel_number]
					else:
						self.thisdict_axis_1[output_file_name] = []
						self.thisdict_axis_1[output_file_name] = self.thisdict_axis_1[output_file_name] + data_cont.data_list_raw[file1].peak_wisths[rep1, channel_number]

			if self.string_y.__contains__("GP") == False and self.string_y.__contains__("Diff") == False:

				if self.string_y.__contains__("prom") == False and self.string_y.__contains__("width") == False:
					str1, str2 = self.string_y.split(" ")
					channel_number = int(str2) - 1

				

					if output_file_name in self.thisdict_axis_2.keys():
						self.thisdict_axis_2[output_file_name] = self.thisdict_axis_2[output_file_name] + data_cont.data_list_raw[file1].peaks[rep1, channel_number]
					else:
						self.thisdict_axis_2[output_file_name] = []
						self.thisdict_axis_2[output_file_name] = self.thisdict_axis_2[output_file_name] + data_cont.data_list_raw[file1].peaks[rep1, channel_number]

				if self.string_y.__contains__("prom") == True:
					str1, str2 = self.string_y.split(" ")
					channel_number = int(str2) - 1

					print (channel_number)

					if output_file_name in self.thisdict_axis_2.keys():
						self.thisdict_axis_2[output_file_name] = self.thisdict_axis_2[output_file_name] + data_cont.data_list_raw[file1].peak_prominences[rep1, channel_number]
					else:
						self.thisdict_axis_2[output_file_name] = []
						self.thisdict_axis_2[output_file_name] = self.thisdict_axis_2[output_file_name] + data_cont.data_list_raw[file1].peak_prominences[rep1, channel_number]

				if self.string_y.__contains__("width") == True:
					str1, str2 = self.string_y.split(" ")
					channel_number = int(str2) - 1

				

					if output_file_name in self.thisdict_axis_2.keys():
						self.thisdict_axis_2[output_file_name] = self.thisdict_axis_2[output_file_name] + data_cont.data_list_raw[file1].peak_widths[rep1, channel_number]
					else:
						self.thisdict_axis_2[output_file_name] = []
						self.thisdict_axis_2[output_file_name] = self.thisdict_axis_2[output_file_name] + data_cont.data_list_raw[file1].peak_wisths[rep1, channel_number]







		
			
		for key in self.thisdict_axis_1.keys():

			self.dot_plot.scatter(self.thisdict_axis_1[key], self.thisdict_axis_2[key], label = key )
			self.dot_plot.legend(loc='upper right')


			self.dens_plot.hist2d(self.thisdict_axis_1[key], self.thisdict_axis_2[key], label = key)

		self.dot_plot.set_ylabel(self.string_x)
		self.dot_plot.set_xlabel(self.string_y)

		self.dens_plot.set_ylabel(self.string_x)
		self.dens_plot.set_xlabel(self.string_y)

		



		line = self.Scale_list.get()

		self.dot_plot.set_xscale (line)
		self.dot_plot.set_yscale (line)

		self.dens_plot.set_xscale (line)
		self.dens_plot.set_yscale (line)

		x = np.array(self.thisdict_axis_1[key])
		y = np.array(self.thisdict_axis_2[key])


		nbins_x = np.sqrt(len(x))
		nbins_y = np.sqrt(len(y))

		sns.kdeplot(x,y, ax = self.dens_plot, shade = True, cmap = "PuBu", cbar = True, cbar_ax = self.colorbar)
		self.dens_plot.set_facecolor('white')


		 
		# Change color palette
		#plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Greens_r)
		#plt.show()




		self.canvas5.draw_idle()

		self.figure5.tight_layout()

	def __init__(self, win_width, win_height, dpi_all):

		self.channel_index = 0
		self.fit_all_flag = False


		self.win_dot_plot = tk.Toplevel()

		self.th_width = round(0.7*self.win_dot_plot.winfo_screenwidth())
		self.th_height = round(0.4*self.win_dot_plot.winfo_screenwidth())

		self.line1 = str(self.th_width) + "x" + str(self.th_height)


		self.win_dot_plot.geometry(self.line1)

		self.frame002 = tk.Frame(self.win_dot_plot)
		self.frame002.pack(side = "left", anchor = "nw")

		self.frame0002 = tk.Frame(self.frame002)
		self.frame0002.pack(side = "top", anchor = "nw")



		self.scrollbar = tk.Scrollbar(self.frame0002)
		self.scrollbar.pack(side = "left", fill = "y")


		self.Datalist = tk.Listbox(self.frame0002, width = 100, height = 10)
		self.Datalist.pack(side = "top", anchor = "nw")

		
		
		
		self.tree = CheckboxTreeview(self.Datalist)
		self.tree.heading("#0",text="Imported datasets",anchor=tk.W)
		self.tree.pack()


		self.tree.config(yscrollcommand = self.scrollbar.set)
		self.scrollbar.config(command = self.tree.yview)

		self.tree.bind('<<TreeviewSelect>>', self.Choose_dataset)



		self.Datalist.config(width = 100, height = 10)

		for i in range(0, len(data_cont.tree_list_name)):
			name = data_cont.tree_list_name[i]
			treetree = d_tree.Data_tree (self.tree, name, data_cont.data_list_raw[i].repetitions)


		self.frame003 = tk.Frame(self.frame002)
		self.frame003.pack(side = "top", anchor = "nw")

		self.frame0003 = tk.Frame(self.frame003)
		self.frame0003.pack(side = "top", anchor = "nw")



		self.frame001 = tk.Frame(self.frame002)
		self.frame001.pack(side = "top", anchor = "nw")

		self.frame000 = tk.Frame(self.win_dot_plot)
		self.frame000.pack(side = "left", anchor = "nw")


		self.figure5 = Figure(figsize=(0.9*self.th_width/dpi_all,0.9*self.th_height/(dpi_all)), dpi = dpi_all)
						
		gs = self.figure5.add_gridspec(1, 21)


		self.dot_plot = self.figure5.add_subplot(gs[0, :10])

		self.dot_plot.set_title("Dot Plot")


		self.dot_plot.set_ylabel('axis 1')
		self.dot_plot.set_xlabel('axis 2')


		self.dens_plot = self.figure5.add_subplot(gs[0, 10:20])

		self.dens_plot.set_title("Density Plot")


		self.dens_plot.set_ylabel('axis 1')
		self.dens_plot.set_xlabel('axis 2')

		self.colorbar = self.figure5.add_subplot(gs[0, 20])

		#self.hist1.set_title("Intensity histogram")

		
		



		self.canvas5 = FigureCanvasTkAgg(self.figure5, self.frame000)
		self.canvas5.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)

		self.toolbar = NavigationToolbar2Tk(self.canvas5, self.frame000)
		self.toolbar.update()
		self.canvas5.get_tk_widget().pack()

		self.figure5.tight_layout()

		self.Export_plot_button = tk.Button(self.frame000, text="Save plot data", command=self.Save_plot_data)
		self.Export_plot_button.pack(side = "top", anchor = "nw")

		

		self.axis_choice = []


		self.Axis_x_label = tk.Label(self.frame001, text = "X axis: ")
		self.Axis_x_label.grid(row = 0, column = 0, sticky = 'ew')

		self.Axis_x_label__choice = ttk.Combobox(self.frame001,values = self.axis_choice,  width = 18 )
		self.Axis_x_label__choice.config(state = "readonly")

		self.Axis_x_label__choice.grid(row = 0, column = 1)

		self.Axis_y_label = tk.Label(self.frame001, text = "Y axis: ")
		self.Axis_y_label.grid(row = 1, column = 0, sticky = 'ew')

		self.Axis_y_label__choice = ttk.Combobox(self.frame001,values = self.axis_choice,  width = 18 )
		self.Axis_y_label__choice.config(state = "readonly")

		self.Axis_y_label__choice.grid(row = 1, column = 1)

		self.Scale_label = tk.Label(self.frame001, text = "Scale: ")
		self.Scale_label.grid(row = 2, column = 0, sticky = 'ew')

		self.Scale_list = ttk.Combobox(self.frame001,values = ["linear", "log"],  width = 18 )
		self.Scale_list.config(state = "readonly")

		self.Scale_list.grid(row = 2, column = 1)

		self.Plot_button = tk.Button(self.frame001, text="Plot", command=self.Plot_dataset)
		self.Plot_button.grid(row = 3, column = 0, columnspan = 2, sticky = 'ew')




		
		self.tree.selection_set(treetree.child_id)