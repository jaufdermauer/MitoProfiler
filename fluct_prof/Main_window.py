#--------------------------
#Importing general modules
#--------------------------
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import filters

import csv

import lmfit

import time

import tifffile
import czifile

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

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

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

from fluct_prof import Functions_sFCS as func


#--------------------------
#End of importing own modules
#--------------------------





class Left_frame :






	def Plot_this_data(self, datasets_pos, rep):

		

		

		self.traces.cla()

		for i in range (datasets_pos.datasets_list[rep].channels_number): 

			if self.channels_flags[datasets_pos.datasets_list[rep].channels_list[i].short_name].get() == 1:

				self.traces.plot(datasets_pos.datasets_list[rep].channels_list[i].fluct_arr.x, datasets_pos.datasets_list[rep].channels_list[i].fluct_arr.y, label = datasets_pos.datasets_list[rep].channels_list[i].short_name)
				self.corr.plot(datasets_pos.datasets_list[rep].channels_list[i].auto_corr_arr.x, datasets_pos.datasets_list[rep].channels_list[i].auto_corr_arr.y, label = datasets_pos.datasets_list[rep].channels_list[i].short_name)

		for i in range (datasets_pos.datasets_list[rep].cross_number):

			if self.channels_flags[datasets_pos.datasets_list[rep].cross_list[i].short_name].get() == 1:

				self.corr.plot(datasets_pos.datasets_list[rep].cross_list[i].cross_corr_arr.x, datasets_pos.datasets_list[rep].cross_list[i].cross_corr_arr.y, label = datasets_pos.datasets_list[rep].cross_list[i].short_name)


		


		
		

		self.traces.set_title("Intensity traces")
		self.traces.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.traces.set_ylabel('Counts (Hz)')
		self.traces.set_xlabel('Time (s)')
		self.traces.legend(loc='upper right')




		self.corr.set_title("Correlation curves")
		self.corr.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.corr.set_ylabel('G(tau)')
		self.corr.set_xlabel('Delay time')
		self.corr.set_xscale ('log')
		self.corr.legend(loc='upper right')


		self.canvas1.draw_idle()

		

		self.figure1.tight_layout()

		
		
	def Continue_Import(self):
		print("Continuing import")
		self.dataset_list = fcs_importer.Fill_datasets_fcs(self.lines)


		for self.dataset in self.dataset_list:

			self.name1 = self.dataset.position + "__" + self.name 

			treetree = d_tree.Data_tree (self.tree, self.name1, self.dataset.repetitions)
			self.tree.selection_set(treetree.child_id)
			data_cont.tree_list.append(treetree)

			data_cont.tree_list_name.append(self.name1)

			data_cont.binning_list.append(1)


			data_cont.data_list_raw.append(self.dataset)


			#data_list_current.append(dataset1)


			data_cont.total_channels_list.append(self.dataset.datasets_list[0].channels_number + self.dataset.datasets_list[0].cross_number)
			data_cont.repetitions_list.append(self.dataset.repetitions)

			data_cont.peaks_list.append([None] * self.dataset.repetitions)

			data_cont.list_of_channel_pairs.append([None])

	def check_positions(self):

		for key in self.checklist.keys():
			if self.checklist[key].get() == 1:
				flag = 1
			else:
				flag = 0

			break


		if flag == 1:

			for key in self.checklist.keys():
				self.checklist[key].set(0)

		if flag == 0:

			for key in self.checklist.keys():
				self.checklist[key].set(1)


	def Import(self):

		if data_cont.initialdirectory == '':
			data_cont.initialdirectory = __file__

		ftypes = [('FCS .fcs', '*.fcs'), ('FCS .SIN', '*.SIN'), ('Text files', '*.txt'), ('All files', '*'), ]
		

		filenames =  tk.filedialog.askopenfilenames(initialdir=os.path.dirname(data_cont.initialdirectory),title = "Select file", filetypes = ftypes)

		
		filename = filenames[0]
		#print (filename)

		line = "file 1 out of " + str(len(filenames))

		self.pb = ttk.Progressbar(self.framepb, orient='horizontal', mode='determinate', length=280)
		self.pb.pack(side = "left", anchor = "nw")
		self.value_label = ttk.Label(self.framepb, text=line)
		self.value_label.pack(side = "left", anchor = "nw")

		for filename_index in range (0, len(filenames)):
			filename = filenames[filename_index]
			if filename != "":

				self.pb['value'] = (filename_index+1)/len(filenames) * 100
				self.value_label['text'] = "file " + str(filename_index + 1) + " out of " + str(len(filenames))

				data_cont.initialdirectory = os.path.dirname(filename)

				

				#progress_window.grab_set()


				self.name = os.path.basename(filename)

				file = codecs.open (filename, encoding='latin')

				self.lines = file.readlines()

				if filename.endswith('.fcs'):

					i = 0

					while i < len(self.lines):



						if self.lines[i].__contains__("CarrierRows"):

							str1 , str2 = self.lines[i].split(' = ')
							CarrierRows = int(str2)

							str1 , str2 = self.lines[i+1].split(' = ')
							CarrierColumns = int(str2)


							break

						i +=1

					self.checklist = {}

					check_button_list = {}

					labels_rows = [None] * CarrierRows
					labels_columns = [None] * CarrierColumns


					if CarrierColumns+CarrierRows > 2:

						self.win_check = tk.Toplevel()

						Label1 = tk.Label(self.win_check, text="Select cells to open: ")
						Label1.grid(row = 0, column = 0, columnspan = CarrierColumns+1, sticky='ew')

						for c in range (0,CarrierColumns):
							labels_columns[c] = tk.Label(self.win_check, text=str(c+1))
							labels_columns[c].grid(row = 1, column = c + 1, sticky='ew')

						for r in range (0,CarrierRows):
							labels_rows[r] = tk.Label(self.win_check, text=chr(r + 65))
							labels_rows[r].grid(row = r + 2, column = 0, sticky='ew')



						for r in range (0,CarrierRows):
							for c in range (0, CarrierColumns):
								self.checklist[r,c] = tk.IntVar(value = 1)

								check_button_list[r,c] = (tk.Checkbutton(self.win_check, variable=self.checklist[r,c]))
								check_button_list[r,c].grid(row = r + 2, column = c + 1, sticky='ew')
									

						
						Button_check_all = tk.Button(self.win_check, text="Check/uncheck all", command=self.check_positions)
						Button_check_all.grid(row = CarrierRows + 2, column = 0, columnspan = CarrierColumns+1, sticky='ew')

						Button_ok = tk.Button(self.win_check, text="OK", command=self.Continue_Import)
						Button_ok.grid(row = CarrierRows + 3, column = 0, columnspan = CarrierColumns+1, sticky='ew')
					
					else:
						self.Continue_Import()


				if filename.endswith('.SIN'): 
					self.dataset = fcs_importer.Fill_datasets_sin(lines)

				#dataset1 = copy.deepcopy(dataset)


				

				#root.update() 



		self.pb.destroy()
		self.value_label.destroy()



	def Select_Unselect(self):

		

		self.Plot_this_data(data_cont.data_list_raw[data_cont.file_index], data_cont.rep_index)

		data_cont.root.update()


	def Plot_data(self, event):

		start = time.time()

		

		index = self.tree.selection()
		num1, num = index[0].split('I')


		

		num = int(num, 16)

		sum1 = num 
		file = 0

		rep = 0
		


		for i in range (len(data_cont.data_list_raw)):
			#print ("I am here")
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




		

		data_cont.file_index = file1-1
		data_cont.rep_index = rep1-1


		

		

		rep = rep1-1


		self.Curve_flags()

		

		self.Plot_this_data(data_cont.data_list_raw[data_cont.file_index], rep)

		#root.update()

	def Delete_dataset(self):
		
		index = self.tree.selection()
		for sel in index:
			self.tree.delete(sel)

	def Delete_all_datasets(self):
		



		for dataset in self.tree.get_children():
			self.tree.delete(dataset)
		self.traces.clear()
		self.corr.clear()
		self.canvas1.draw_idle()
	
		self.figure1.tight_layout()
	

		data_list_raw = []
		data_list_current = []
		tree_list = []
		tree_list_name = []


	def Curve_flags(self):

		self.frame0003.destroy()

		self.frame0003 = tk.Frame(self.frame024)
		self.frame0003.pack(side = "left", anchor = "nw")

		self.flags_dict = {}
		self.channels_flags = {}
		self.cross_flags = []
		column_counter = 0

		channels_to_display = 0

		for i in range (len(data_cont.data_list_raw)):
			if data_cont.data_list_raw[i].datasets_list[0].channels_number > channels_to_display:
				channels_to_display = data_cont.data_list_raw[i].datasets_list[0].channels_number
				file_index_local = i


		for item in data_cont.data_list_raw[file_index_local].datasets_list[data_cont.rep_index].channels_list:
			str1, str2 = item.short_name.split(" ")
			very_short_name = "ch0" + str2
			self.channels_flags[item.short_name] = tk.IntVar(value=1)
			self.flags_dict[item.short_name] = tk.Checkbutton(self.frame0003, text=very_short_name, variable=self.channels_flags[item.short_name], command = self.Select_Unselect)
			self.flags_dict[item.short_name].grid(row = 0, column = column_counter, sticky='w')
			column_counter +=1

		for item in data_cont.data_list_raw[file_index_local].datasets_list[data_cont.rep_index].cross_list:
			str1, str2 = item.short_name.split(" vs ")
			strs = str1.split(" ")
			very_short_name = "ch" + strs[1] + str2
			self.channels_flags[item.short_name] = tk.IntVar(value=1)
			self.flags_dict[item.short_name] = tk.Checkbutton(self.frame0003, text=very_short_name, variable=self.channels_flags[item.short_name], command = self.Select_Unselect)
			self.flags_dict[item.short_name].grid(row = 0, column = column_counter, sticky='w')
			column_counter +=1


	def __init__ (self, frame0, win_width, win_height, dpi_all):



		pixel = tk.PhotoImage(width=1, height=1)


		

		self.frame01 = tk.Frame(frame0)
		self.frame01.pack(side="top", fill="x")


		self.Import_Button = tk.Button(self.frame01, text="Import", command=self.Import)
		self.Import_Button.pack(side = "left", anchor = "nw")

		self.Clear_Button = tk.Button(self.frame01, text="Delete dataset", command=self.Delete_dataset)
		self.Clear_Button.pack(side = "left", anchor = "nw")

		self.Clear_all_Button = tk.Button(self.frame01, text="Delete all", command=self.Delete_all_datasets)
		self.Clear_all_Button.pack(side = "left", anchor = "nw")


		self.frame02 = tk.Frame(frame0)
		self.frame02.pack(side="left", fill="x", anchor = "nw")

		self.frame04 = tk.Frame(frame0)
		self.frame04.pack(side="left", fill="x", anchor = "nw")


		self.frame03 = tk.Frame(self.frame02)
		self.frame03.pack(side="top", fill="x")



		self.scrollbar = tk.Scrollbar(self.frame03)
		self.scrollbar.pack(side = "left", fill = "y")


		self.Datalist = tk.Listbox(self.frame03, width = 150, height = 10)
		self.Datalist.pack(side = "left", anchor = "nw")
		
		
		
		self.tree=CheckboxTreeview(self.Datalist)
		self.tree.heading("#0",text="Imported datasets",anchor=tk.W)
		self.tree.pack()


		self.tree.config(yscrollcommand = self.scrollbar.set)
		self.scrollbar.config(command = self.tree.yview)

		self.tree.bind('<<TreeviewSelect>>', self.Plot_data)

		self.Datalist.config(width = 100, height = 10)

		self.frame024 = tk.Frame(self.frame02)
		self.frame024.pack(side = "top", fill = "x", anchor='nw')

		self.frame0003 = tk.Frame(self.frame024)
		self.frame0003.pack(side = "left", fill = "x")


		#self.chkbtn = tk.Checkbutton(self.frame0003, text="ch1", variable=1, command=Norm)
		#self.chkbtn.grid(row = 0, column = 0, sticky='w')

		self.frame023 = tk.Frame(self.frame02)
		self.frame023.pack(side="left", fill="x")


		self.Restruct_button = tk.Button(self.frame023, text="Restructure data", command=fun.Restruct_fun)
		self.Restruct_button.grid(row = 0, column = 0, sticky="EW")

		self.Threshold_button = tk.Button(self.frame023, text="Peak analysis", command=fun.Threshold_fun)
		self.Threshold_button.grid(row = 1, column = 0, sticky="EW")

		self.Diffusion_button = tk.Button(self.frame023, text="Diffusion analysis", command=fun.Diffusion_fun)
		self.Diffusion_button.grid(row = 2, column = 0, sticky="EW")

		self.Add_to_plot_button = tk.Button(self.frame023, text="Plot", command=fun.Which_tab)
		self.Add_to_plot_button.grid(row = 3, column = 0, sticky="EW")

		
		self.Add_to_plot_button = tk.Button(self.frame023, text="Dot Plot", command=fun.Dot_Plot_fun)
		self.Add_to_plot_button.grid(row = 4, column = 0, sticky="EW")

		self.Output_button = tk.Button(self.frame023, text="Output", command=fun.Export_function)
		self.Output_button.grid(row = 5, column = 0, sticky="EW")

		self.figure1 = Figure(figsize=(0.85*win_height/dpi_all,0.85*win_height/dpi_all), dpi = dpi_all)




		gs = self.figure1.add_gridspec(3, 2)


		self.traces = self.figure1.add_subplot(gs[:1, :2])

		self.traces.set_title("Intensity traces")

		self.traces.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.traces.set_ylabel('intensity (a.u.)')
		self.traces.set_xlabel('Time (s)')

		


		self.corr = self.figure1.add_subplot(gs[1, :2])

		self.corr.set_title("Correlation curves")

		self.corr.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.corr.set_ylabel('G (tau)')
		self.corr.set_xlabel('Delay time')


		self.diff_plot = self.figure1.add_subplot(gs[2, 0])

		self.diff_plot.set_title("Diffusion")
		self.diff_plot.set_ylabel('Diff. Coeff.')
		



		self.gp_plot = self.figure1.add_subplot(gs[2, 1])

		self.gp_plot.set_title("General Polarization")
		self.gp_plot.set_ylabel('GP')





		self.canvas1 = FigureCanvasTkAgg(self.figure1, self.frame04)
		self.canvas1.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)

		self.toolbar = NavigationToolbar2Tk(self.canvas1, self.frame04)
		self.toolbar.update()
		self.canvas1.get_tk_widget().pack()

		self.figure1.tight_layout()

		self.framepb = tk.Frame(frame0)
		self.framepb.pack(side="top", fill="x")



class sFCS_frame:

	def __init__ (self):
		self.channels_to_display = []
		self.list_of_y = []
		self.array_length = 0
		self.channels_number = 0
		self.n_lines = 0

	def first_degree_bleaching(self, x, a, b):
		return a*x+b
	
	def polynomial_bleaching(self, x, a,b,c,d,e):
		return a*x**4 + b*x**3 + c*x**2 + d*x + e
	
	def Transfer_extracted(self):
		name = self.dataset_names [self.file_number]
		if name+"1" in self.dictionary_of_extracted:
			newname = name+"1"
			dataset = self.dictionary_of_extracted[newname]
			treetree = d_tree.Data_tree (data_cont.data_frame.tree, newname, dataset.repetitions)
			data_cont.tree_list.append(treetree)
			data_cont.tree_list_name.append(newname)
			data_cont.binning_list.append(1)
			data_cont.data_list_raw.append(dataset)
			#data_list_current.append(dataset1)
			data_cont.total_channels_list.append(dataset.datasets_list[0].channels_number + dataset.datasets_list[0].cross_number)
			data_cont.repetitions_list.append(dataset.repetitions)
			data_cont.peaks_list.append([None] * dataset.repetitions)
			data_cont.list_of_channel_pairs.append([None])
		if name+"2" in self.dictionary_of_extracted:
			newname = name+"2"
			dataset = self.dictionary_of_extracted[newname]
			treetree = d_tree.Data_tree (data_cont.data_frame.tree, newname, dataset.repetitions)
			data_cont.tree_list.append(treetree)
			data_cont.tree_list_name.append(newname)
			data_cont.binning_list.append(1)
			data_cont.data_list_raw.append(dataset)
			#data_list_current.append(dataset1)
			data_cont.total_channels_list.append(dataset.datasets_list[0].channels_number + dataset.datasets_list[0].cross_number)
			data_cont.repetitions_list.append(dataset.repetitions)
			data_cont.peaks_list.append([None] * dataset.repetitions)
			data_cont.list_of_channel_pairs.append([None])
		if name+"cross" in self.dictionary_of_extracted:
			newname = name+"cross"
			dataset = self.dictionary_of_extracted[newname]
			treetree = d_tree.Data_tree (data_cont.data_frame.tree, newname, dataset.repetitions)
			data_cont.tree_list.append(treetree)
			data_cont.tree_list_name.append(newname)
			data_cont.binning_list.append(1)
			data_cont.data_list_raw.append(dataset)
			#data_list_current.append(dataset1)
			data_cont.total_channels_list.append(dataset.datasets_list[0].channels_number + dataset.datasets_list[0].cross_number)
			data_cont.repetitions_list.append(dataset.repetitions)
			data_cont.peaks_list.append([None] * dataset.repetitions)
			data_cont.list_of_channel_pairs.append([None])

	def next_channel(self):
		if int(self.Rep_Display__choice.get()) < len(self.reps_to_display):
			self.Rep_Display__choice.set(str(int(self.Rep_Display__choice.get())+1))
			self.Plot_this_file()

	def Plot_this_file(self):

		self.traces.cla()
		self.corr.cla()


		name = self.dataset_names [self.file_number]
		if name+"1" in self.dictionary_of_extracted:
			newname=name+"1"
			rep = int(self.Rep_Display__choice.get()) - 1
			channel = self.Chan_Display__choice.get()
			dataset = self.dictionary_of_extracted [newname] 
			if channel == 'all' and (self.Line_Display__choice.get() == "line 1" or  self.Line_Display__choice.get() == "all"):
				names = [] #avoid doubles
				for i in range (dataset.datasets_list[rep].channels_number): 
					current_channels_list = dataset.datasets_list[rep].channels_list[i]
					if names.count(current_channels_list.short_name) == 0:
						names.append(current_channels_list.short_name)
						popt, pcov = curve_fit(self.polynomial_bleaching, current_channels_list.fluct_arr.x, current_channels_list.fluct_arr.y)
						self.traces.plot(current_channels_list.fluct_arr.x, current_channels_list.fluct_arr.y, label = current_channels_list.short_name)
						self.traces.plot(current_channels_list.fluct_arr.x, self.polynomial_bleaching(np.array(current_channels_list.fluct_arr.x, dtype = np.float64), *popt), label = current_channels_list.short_name + " bleaching / OOF")
						self.corr.plot(current_channels_list.auto_corr_arr.x, current_channels_list.auto_corr_arr.y, label = current_channels_list.short_name)
				for i in range (0, dataset.datasets_list[rep].cross_number):
						if names.count(dataset.datasets_list[rep].cross_list[i].short_name) == 0:
							self.corr.plot(dataset.datasets_list[rep].cross_list[i].cross_corr_arr.x, dataset.datasets_list[rep].cross_list[i].cross_corr_arr.y, label = dataset.datasets_list[rep].cross_list[i].short_name)
		
		if name+"2" in self.dictionary_of_extracted:
			newname = name+"2"
			rep = int(self.Rep_Display__choice.get()) - 1
			channel = self.Chan_Display__choice.get()
			dataset = self.dictionary_of_extracted [newname] 
			if channel == 'all' and (self.Line_Display__choice.get() == "line 2" or  self.Line_Display__choice.get() == "all"):

				for i in range (0, dataset.datasets_list[rep].channels_number): 
					current_channels_list = dataset.datasets_list[rep].channels_list[i]
					popt, pcov = curve_fit(self.polynomial_bleaching, current_channels_list.fluct_arr.x, current_channels_list.fluct_arr.y)
					#self.traces2.plot(current_channels_list.fluct_arr.x, current_channels_list.fluct_arr.y, label = current_channels_list.short_name)
					#self.traces2.plot(current_channels_list.fluct_arr.x, self.polynomial_bleaching(np.array(current_channels_list.fluct_arr.x, dtype = np.float64), *popt), label = current_channels_list.short_name + " bleaching / OOF")
					self.corr.plot(current_channels_list.auto_corr_arr.x, current_channels_list.auto_corr_arr.y, label = current_channels_list.short_name)
				for i in range (0, dataset.datasets_list[rep].cross_number):
					self.corr.plot(dataset.datasets_list[rep].cross_list[i].cross_corr_arr.x, dataset.datasets_list[rep].cross_list[i].cross_corr_arr.y, label = dataset.datasets_list[rep].cross_list[i].short_name)



		if name+"cross" in self.dictionary_of_extracted:
			newname = name+"cross"
			rep = int(self.Rep_Display__choice.get()) - 1
			channel = self.Chan_Display__choice.get()
			dataset = self.dictionary_of_extracted [newname] 
			if channel == 'all':
				for i in range (0, dataset.datasets_list[rep].cross_number):
					self.corr.plot(dataset.datasets_list[rep].cross_list[i].cross_corr_arr.x, dataset.datasets_list[rep].cross_list[i].cross_corr_arr.y, label = dataset.datasets_list[rep].cross_list[i].short_name)



		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self.traces.set_title("Intensity traces")
			self.traces.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
			self.traces.set_ylabel('Counts (Hz)')
			self.traces.set_xlabel('Time (s)')
			self.traces.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)




			self.corr.set_title("Correlation curves")
			self.corr.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
			self.corr.set_ylabel('G(tau)')
			self.corr.set_xlabel('Delay time')
			self.corr.set_xscale ('log')
			self.corr.legend(loc='upper right')

		self.canvas1.draw_idle()

		

		self.figure1.tight_layout()
		

	def Empty_function(self):
		print("Empty function invoked")

	def Extract_trace(self):

		if self.Scantype__choice.get() == "1 focus":
			sedec = Sidecut_sFCS(self.dataset_list[self.file_number])
			self.n_lines = 1
			if len(sedec.array.shape) == 3:
				self.channels_number = sedec.array.shape[0]
				self.array_length = sedec.array.shape[1]
			else:
				self.channels_number = 1
				self.array_length = sedec.array.shape[0]
		elif self.Scantype__choice.get() == "2 focus":
			self.n_lines = 2
			sedec = Sidecut_2fsFCS(self.dataset_list[self.file_number])
			if len(sedec.array.shape) == 3: #1 color
				self.channels_number = 1
				self.array_length = sedec.array.shape[0]
			if len(sedec.array.shape) == 4: #2 color
				self.channels_number = 2
				self.array_length = sedec.array.shape[1]

		print("shape ", sedec.array.shape)

		print("channels number ", self.channels_number)


		bins = int(self.Binning__choice.get())
		lower_lim = int(self.borders_entry.get())
		upper_lim = int(self.borders_entry_end.get())
		t_lower = int(self.time_entry.get())
		t_upper = int(self.time_entry_end.get())

		#print(sedec.array.shape[0])

		self.channels_to_display = []
		self.list_of_y = []
		counter_of_invalid_channels = 0
		for channel_no in range(0, self.channels_number):

			#try:

			y = sedec.isolate_maxima(channel_no, bins, lower_lim, upper_lim, t_lower, t_upper)
			#print("y shape", y.shape)


			self.channels_to_display.append(str(channel_no))
			self.list_of_y.append(y)

			#except:

				#print ("channel number ", channel_no, " has not enough signal")
				#counter_of_invalid_channels+=1


			self.channels_number-=counter_of_invalid_channels

			#self.traces.plot(x, y, label = "channel " + str(channel_no))

			#self.canvas1.draw_idle()

		

			#self.figure1.tight_layout()


		self.channels_to_display.append('all')
		self.Chan_Display__choice.config(values = self.channels_to_display)
		
	def correlate(self, bleaching_correction = True):
		
		name = self.dataset_names [self.file_number]
		timestep = float(self.Timestep_entry.get())
		repetitions = int(self.Repetitions_entry.get())
		self.reps_to_display = []
		for i in range (0, repetitions):
			self.reps_to_display.append(i+1)
				
		x_full = np.linspace(0, self.array_length*timestep, num=self.array_length)
		
		length_rep = int (self.array_length/repetitions)
		
		dataset_list_arg1 = []
		dataset_list_arg2 = []
		dataset_list_arg_cross = []
		#autocorrelation
		for rep_index_i in range (repetitions):

			lines_list_arg = []

			for l in range(self.n_lines):

				channels_list_arg = []

				for channel in range (self.channels_number):

					end = length_rep*(rep_index_i + 1)
					start = end - length_rep

					#print(channel, rep_index_i, start, end)

					if rep_index_i == repetitions-1:

						if end != len (x_full) - 1:

							end = len (x_full) - 1

					x = x_full[start : end]
					
					min1 = min(x)

					x1 = [a - min1 for a in x]

					x = x1

					print("start/end ", start, end)

					if self.Scantype__choice.get() == "1 focus":
						y = self.list_of_y[channel][start : end]

					elif self.Scantype__choice.get() == "2 focus":
						y = self.list_of_y[channel][l][start : end]

					if(bleaching_correction):
						popt, pcov = curve_fit(self.polynomial_bleaching, x, y)
						print("bleaching parameters: ", popt)
						y_bc = []	#bleaching corrected y
						for i,ys in enumerate(y):
							correction_factor = np.sqrt(self.polynomial_bleaching(x[i], *popt)/self.polynomial_bleaching(0, *popt))
							#print(correction_factor)
							y_bc.append(ys/correction_factor+self.polynomial_bleaching(0, *popt)*(1-correction_factor))

						Tr = fcs_importer.XY_plot(x,y_bc)
					else:
						Tr = fcs_importer.XY_plot(x,y)

					timestep = x[1] - x[0]

					x1, y1 = corr_py.correlate_full (timestep, np.array(Tr.y), np.array(Tr.y))

					AutoCorr = fcs_importer.XY_plot(x1,y1)

					long_name = name + "channel " + str(channel)

					short_name = "channel " + str(channel)

					Ch_dataset = fcs_importer.fcs_channel (long_name, Tr, AutoCorr, short_name)

					channels_list_arg.append(Ch_dataset)
					
				lines_list_arg.append(channels_list_arg)

			lines_cross_list_arg = []

			#cross correlation between channels
			for l in range(self.n_lines):

				cross_list_arg = []

				if self.channels_number > 1:
					channel1 = 0
					while channel1 < self.channels_number-1:
					#for channel1 in range (0, self.channels_number):
						end = length_rep*(rep_index_i + 1)
						start = end - length_rep

						#print(channel, rep_index_i, start, end)

						if rep_index_i == repetitions-1:

							if end != len (x_full) - 1:

								end = len (x_full) - 1

						

						x = x_full[start : end]

						if self.Scantype__choice.get() == "1 focus":
							y = self.list_of_y[channel1][start : end]

						elif self.Scantype__choice.get() == "2 focus":
							y = self.list_of_y[channel1][l][start : end]

						#print("cc line ", l, "ch1", y)

						min1 = min(x)

						x1 = [a - min1 for a in x]

						x = x1

						if(bleaching_correction):
							popt, pcov = curve_fit(self.polynomial_bleaching, x, y)
							print(popt)
							y_bc = []	#bleaching corrected y
							for i,ys in enumerate(y):
								correction_factor = np.sqrt(self.polynomial_bleaching(x[i], *popt)/self.polynomial_bleaching(0, *popt))
								#print(correction_factor)
								y_bc.append(ys/correction_factor+self.polynomial_bleaching(0, *popt)*(1-correction_factor))

							Tr1 = fcs_importer.XY_plot(x,y_bc)
						else:
							Tr1 = fcs_importer.XY_plot(x,y)

						channel2 = channel1 + 1
						while channel2 < self.channels_number:

						#for channel2 in range (channel1 +1, self.channels_number):
							end = length_rep*(rep_index_i + 1)
							start = end - length_rep

							#print(channel, rep_index_i, start, end)

							if rep_index_i == repetitions-1:

								if end != len (x_full) - 1:

									end = len (x_full) - 1

							

							x = x_full[start : end]

							if self.Scantype__choice.get() == "1 focus":
								y = self.list_of_y[channel2][start : end]

							elif self.Scantype__choice.get() == "2 focus":
								y = self.list_of_y[channel2][l][start : end]



							min1 = min(x)

							x1 = [a - min1 for a in x]

							x = x1

							#print("cc line ", l, "ch2", y)

							if(bleaching_correction):
								popt, pcov = curve_fit(self.polynomial_bleaching, x, y)
								print(popt)
								y_bc = []	#bleaching corrected y
								for i,ys in enumerate(y):
									correction_factor = np.sqrt(self.polynomial_bleaching(x[i], *popt)/self.polynomial_bleaching(0, *popt))
									#print(correction_factor)
									y_bc.append(ys/correction_factor+self.polynomial_bleaching(0, *popt)*(1-correction_factor))

								Tr2 = fcs_importer.XY_plot(x,y_bc)
							else:
								Tr2 = fcs_importer.XY_plot(x,y)


							Tr2 = fcs_importer.XY_plot(x,y)

							timestep = Tr1.x[1] - Tr1.x[0]

							x1, y1 = corr_py.correlate_full (timestep, np.array(Tr1.y), np.array(Tr2.y))

							CrossCorr_12 = fcs_importer.XY_plot(x1,y1)

							short_name_12 = "channel " + str(channel1) + " vs " + "channel " + str(channel2)

							x1, y1 = corr_py.correlate_full (timestep, np.array(Tr2.y), np.array(Tr1.y))

							CrossCorr_21 = fcs_importer.XY_plot(x1,y1)

							short_name_21 = "channel " + str(channel2) + " vs " + "channel " + str(channel1)

							Cross_dataset = fcs_importer.fcs_cross (short_name_12, CrossCorr_12, short_name_12)
							cross_list_arg.append(Cross_dataset)
							Cross_dataset = fcs_importer.fcs_cross (short_name_21, CrossCorr_21, short_name_21)
							cross_list_arg.append(Cross_dataset)

							channel2 += 1

						channel1 += 1

				lines_cross_list_arg.append(cross_list_arg)
			self.Rep_Display__choice.set(str(1))

			############# CROSS CORRELATION BETWEEN LINES FOR EACH CHANNEL AND EACH REPETITION ####################

			if self.Scantype__choice.get() == "2 focus":
				cross_list_arg = []
				if self.channels_number > 1:
					channel = 0
					while channel < self.channels_number:
					#for channel1 in range (0, self.channels_number):
						end = length_rep*(rep_index_i + 1)
						start = end - length_rep

						#print(channel, rep_index_i, start, end)

						if rep_index_i == repetitions-1:

							if end != len (x_full) - 1:

								end = len (x_full) - 1

						

						x = x_full[start : end]
						y = self.list_of_y[channel][0][start : end]

						#print(y)

						min1 = min(x)

						x1 = [a - min1 for a in x]

						x = x1

						Tr1 = fcs_importer.XY_plot(x,y)

						y = self.list_of_y[channel][1][start : end]

						min1 = min(x)

						x1 = [a - min1 for a in x]

						x = x1

						

						Tr2 = fcs_importer.XY_plot(x,y)

						timestep = Tr1.x[1] - Tr1.x[0]

						x1, y1 = corr_py.correlate_full (timestep, np.array(Tr1.y), np.array(Tr2.y))

						CrossCorr_12 = fcs_importer.XY_plot(x1,y1)

						short_name_12 = "channel " + str(channel) + " line 1" +  " vs " + "line 2"

						x1, y1 = corr_py.correlate_full (timestep, np.array(Tr2.y), np.array(Tr1.y))

						CrossCorr_21 = fcs_importer.XY_plot(x1,y1)

						short_name_21 = "channel " + str(channel) + " line 2" +  " vs " + "line 1"

						Cross_dataset = fcs_importer.fcs_cross (short_name_12, CrossCorr_12, short_name_12)
						cross_list_arg.append(Cross_dataset)
						Cross_dataset = fcs_importer.fcs_cross (short_name_21, CrossCorr_21, short_name_21)
						cross_list_arg.append(Cross_dataset)

						channel += 1
				lines_cross_list_arg.append(cross_list_arg)
			####END#####



			print(len(channels_list_arg), len(cross_list_arg), len(lines_cross_list_arg))
			
			#for 2fsFCCS:
			#lines_cross_list_arg[0]: CC line 1
			#lines_cross_list_arg[1]: CC line 2
			#lines_cross_list_arg[2]: CC between lines

			FCS_Dataset1 =  fcs_importer.Dataset_fcs(self.channels_number, len(lines_cross_list_arg[0]), lines_list_arg[0], lines_cross_list_arg[0])
			dataset_list_arg1.append(FCS_Dataset1)
			dataset1 = 	fcs_importer.Full_dataset_fcs(repetitions, dataset_list_arg1)
			self.dictionary_of_extracted [name+"1"] = dataset1
			
			#different datasets for 2 lines
			if self.Scantype__choice.get() == "2 focus":
				FCS_Dataset2 =  fcs_importer.Dataset_fcs(self.channels_number, len(lines_cross_list_arg[1]), lines_list_arg[1], lines_cross_list_arg[1])
				FCS_Dataset_cross =  fcs_importer.Dataset_fcs(self.channels_number, len(lines_cross_list_arg[2]), lines_list_arg[0], lines_cross_list_arg[2])
				dataset_list_arg2.append(FCS_Dataset2)
				dataset_list_arg_cross.append(FCS_Dataset_cross)
				dataset2 = 	fcs_importer.Full_dataset_fcs(repetitions, dataset_list_arg2)
				dataset_cross = 	fcs_importer.Full_dataset_fcs(repetitions, dataset_list_arg_cross)
				self.dictionary_of_extracted [name+"2"] = dataset2
				self.dictionary_of_extracted [name+"cross"] = dataset_cross
		
		self.Rep_Display__choice.config(values = self.reps_to_display)

		self.Plot_this_file()

		#name = data_cont.tree_list_name[data_cont.file_index] + " " + str(repetitions_new)

		#treetree = d_tree.Data_tree (self.tree, name, dataset.repetitions)

		#treetree = d_tree.Data_tree (data_cont.data_frame.tree, name, dataset.repetitions)

		#data_cont.tree_list.append(treetree)

		#data_cont.tree_list_name.append(name)

		#data_cont.binning_list.append(1)


		#data_cont.data_list_raw.append(dataset)


		#data_list_current.append(dataset1)


		#data_cont.total_channels_list.append(dataset.datasets_list[0].self.channels_number + dataset.datasets_list[0].cross_number)
		#data_cont.repetitions_list.append(dataset.repetitions)

		#data_cont.peaks_list.append([None] * dataset.repetitions)

		#data_cont.list_of_channel_pairs.append([None])
		

	def Tree_selection(self, event):

		index = self.tree.selection()
		num1, num = index[0].split('I')


		

		num = int(num, 16)-1

		self.file_number = num

		if self.Scantype__choice.get() == "1 focus":
			eg= func.File_sFCS(self.dataset_list[self.file_number])

			bins = 1
			slices = 1

			binned_data = eg.intensity_carpet_plot(1, bin_size=bins, n_slices = slices)
			binned_data2 = eg.intensity_carpet_plot(0, bin_size=bins, n_slices = slices)
			
			self.time_entry_end.insert("end", str(len(binned_data[0])))
			bdf = binned_data.flatten()		#flatten to find max/min in list
			val = filters.threshold_otsu(binned_data)	#otsu threshold to compensate for spikes in intensity
			self.image.grid(False)	#deactivate grid
			self.image2.grid(False)	#deactivate grid
			self.image.imshow(binned_data[:,0:10000],origin="lower", cmap = "rainbow", vmin=min(bdf), vmax=(max(bdf)+val)/2)
			self.image2.imshow(binned_data2[:,0:10000],origin="lower", cmap = "rainbow", vmin=min(bdf), vmax=(max(bdf)+val)/2)
			self.canvas1.draw_idle()
		
		elif self.Scantype__choice.get() == "2 focus":
			eg= func.File_2fsFCS(self.dataset_list[self.file_number])

			bins = 1
			slices = 1

			binned_data = eg.intensity_carpet_plot(1, 0, bin_size=bins, n_slices = slices)
			binned_data2 = eg.intensity_carpet_plot(1, 1, bin_size=bins, n_slices = slices)
			bdf = binned_data.flatten()		#flatten to find max/min in list
			val = filters.threshold_otsu(binned_data)	#otsu threshold to compensate for spikes in intensity
			self.image.grid(False)	#deactivate grid
			self.image2.grid(False)	#deactivate grid
			truncated_binned_data = np.array(np.array(binned_data).T.tolist())[0:32*len(binned_data)].T.tolist()
			truncated_binned_data2 = np.array(np.array(binned_data2).T.tolist())[0:32*len(binned_data)].T.tolist()
			self.image.imshow(truncated_binned_data,origin="lower", cmap = "rainbow", vmin=min(bdf), vmax=(max(bdf)+val)/2)
			self.image2.imshow(truncated_binned_data2,origin="lower", cmap = "rainbow", vmin=min(bdf), vmax=(max(bdf)+val)/2)

			#self.image.set_xscale('log')
			#self.image2.set_xscale('log')

			self.canvas1.draw_idle()

		self.figure1.tight_layout()

		self.Plot_this_file()
        		




	def Import(self):

		if data_cont.initialdirectory == '':
			data_cont.initialdirectory = __file__

		ftypes = [('CZI .czi', '*.czi'),('LSM .lsm', '*.lsm'), ('Tif .tif', '*.tif'), ('All files', '*'), ]
		

		filenames =  tk.filedialog.askopenfilenames(initialdir=os.path.dirname(data_cont.initialdirectory),title = "Select file", filetypes = ftypes)

		for filename_index in range (0, len(filenames)):
			filename = filenames[filename_index]
			if filename != "":

				data_cont.initialdirectory = os.path.dirname(filename)


				self.name = os.path.basename(filename)

				self.dataset_list.append(filename)

				self.dataset_names.append(self.name)

				treetree = d_tree.Data_tree (self.tree, self.name, 0)
				self.tree.selection_set(treetree.child_id)
			

		

			

	def __init__ (self, frame0, win_width, win_height, dpi_all):

		self.dictionary_of_extracted = {}


		self.dataset_list = []
		self.dataset_names = []
		self.file_number = 0



		pixel = tk.PhotoImage(width=1, height=1)


		

		self.frame01 = tk.Frame(frame0)
		self.frame01.pack(side="top", fill="x")


		self.Import_Button = tk.Button(self.frame01, text="Import", command=self.Import)
		self.Import_Button.pack(side = "left", anchor = "nw")

		self.Clear_Button = tk.Button(self.frame01, text="Delete dataset", command=self.Empty_function)
		self.Clear_Button.pack(side = "left", anchor = "nw")

		self.Clear_all_Button = tk.Button(self.frame01, text="Delete all", command=self.Empty_function)
		self.Clear_all_Button.pack(side = "left", anchor = "nw")


		self.frame02 = tk.Frame(frame0)
		self.frame02.pack(side="left", fill="x", anchor = "nw")

		self.frame04 = tk.Frame(frame0)
		self.frame04.pack(side="left", fill="x", anchor = "nw")


		self.frame03 = tk.Frame(self.frame02)
		self.frame03.pack(side="top", fill="x")



		self.scrollbar = tk.Scrollbar(self.frame03)
		self.scrollbar.pack(side = "left", fill = "y")


		self.Datalist = tk.Listbox(self.frame03, width = 150, height = 10)
		self.Datalist.pack(side = "left", anchor = "nw")
		
		
		
		self.tree=CheckboxTreeview(self.Datalist)
		self.tree.heading("#0",text="Imported datasets",anchor=tk.W)
		self.tree.pack()


		self.tree.config(yscrollcommand = self.scrollbar.set)
		self.scrollbar.config(command = self.tree.yview)

		self.tree.bind('<<TreeviewSelect>>', self.Tree_selection)

		self.Datalist.config(width = 100, height = 10)

		self.frame024 = tk.Frame(self.frame02)
		self.frame024.pack(side = "top", fill = "x", anchor='nw')

		self.frame0003 = tk.Frame(self.frame024)
		self.frame0003.pack(side = "left", fill = "x")


		#self.chkbtn = tk.Checkbutton(self.frame0003, text="ch1", variable=1, command=Norm)
		#self.chkbtn.grid(row = 0, column = 0, sticky='w')

		self.frame023 = tk.Frame(self.frame02)
		self.frame023.pack(side="left", fill="x")

		gridrow = 0

		self.Extract_button = tk.Button(self.frame023, text="Extract trace", command=self.Extract_trace)
		self.Extract_button.grid(row = gridrow, column = 0, columnspan = 2, sticky="EW")
		gridrow += 1
		
		"""
		self.Correlate_button = tk.Button(self.frame023, text="Correlate no bleaching", command=lambda: self.correlate(False))
		self.Correlate_button.grid(row = gridrow, column = 0, columnspan = 2, sticky="EW")
		gridrow += 1
		"""
		self.Binning_label = tk.Label(self.frame023,  text = "Pixel binning: ")
		self.Binning_label.grid(row = gridrow, column = 0, sticky = 'ew')

		self.Binning__choice = ttk.Combobox(self.frame023,values = ["0","1","2","3","4","5","6", "gaussian"],  width = 10)
		self.Binning__choice.config(state = "readonly")
		self.Binning__choice.grid(row = gridrow, column = 1, sticky = 'ew')
		self.Binning__choice.set("3")
		gridrow += 1

		
		self.borders_label = tk.Label(self.frame023,  text = "Borders from: ", width = 9)
		self.borders_label.grid(row = gridrow, column = 0, sticky = 'ew')

		self.borders_entry = tk.Entry(self.frame023, width = 9)
		self.borders_entry.grid(row = gridrow, column = 1, sticky='ew')
		self.borders_entry.insert("end", str(3))
		gridrow += 1

		self.borders_label_end = tk.Label(self.frame023,  text = "to: ", width = 9)
		self.borders_label_end.grid(row = gridrow, column = 0, sticky = 'ew')

		self.borders_entry_end = tk.Entry(self.frame023, width = 9)
		self.borders_entry_end.grid(row = gridrow, column = 1, sticky='ew')
		self.borders_entry_end.insert("end", str(125))
		gridrow += 1

		self.time_label = tk.Label(self.frame023,  text = "Analyze from: ", width = 9)
		self.time_label.grid(row = gridrow, column = 0, sticky = 'ew')

		self.time_entry = tk.Entry(self.frame023, width = 9)
		self.time_entry.grid(row = gridrow, column = 1, sticky='ew')
		self.time_entry.insert("end", str(0))
		gridrow += 1

		self.time_label_end = tk.Label(self.frame023,  text = "to: ", width = 9)
		self.time_label_end.grid(row = gridrow, column = 0, sticky = 'ew')

		self.time_entry_end = tk.Entry(self.frame023, width = 9)
		self.time_entry_end.grid(row = gridrow, column = 1, sticky='ew')
		gridrow += 1
		

		self.Repetitions_label = tk.Label(self.frame023,  text = "Repetitions: ")
		self.Repetitions_label.grid(row = gridrow, column = 0, sticky = 'ew')

		self.Repetitions_entry = tk.Entry(self.frame023, width = 9)
		self.Repetitions_entry.grid(row = gridrow, column = 1, sticky='ew')
		self.Repetitions_entry.insert("end", str(1))
		gridrow += 1

		self.Timestep_label = tk.Label(self.frame023,  text = "Timestep: ")
		self.Timestep_label.grid(row = gridrow, column = 0, sticky = 'ew')

		self.Timestep_entry = tk.Entry(self.frame023, width = 9)
		self.Timestep_entry.grid(row = gridrow, column = 1, sticky='ew')
		self.Timestep_entry.insert("end", str(0.00059))
		gridrow += 1

		self.Bleaching_button = tk.Button(self.frame023, text="Correlate bleaching", command=lambda: self.correlate(True))
		self.Bleaching_button.grid(row = gridrow, column = 0, columnspan = 2, sticky="EW")
		gridrow += 1

		self.Display_label = tk.Label(self.frame023,  text = "Display: ")
		self.Display_label.grid(row = gridrow, column = 0, columnspan = 2, sticky = 'w')
		gridrow += 1

		self.Rep_Display_label = tk.Label(self.frame023,  text = "Repetition: ")
		self.Rep_Display_label.grid(row = gridrow, column = 0, sticky = 'ew')

		self.Rep_Display__choice = ttk.Combobox(self.frame023,values = ["1","2","3"],  width = 18 )
		self.Rep_Display__choice.config(state = "readonly")
		self.Rep_Display__choice.grid(row = gridrow, column = 1, sticky = 'ew')
		self.Rep_Display__choice.set("1")
		gridrow += 1

		self.Next_Channel_button = tk.Button(self.frame023, text="display next channel", command=lambda: self.next_channel())
		self.Next_Channel_button.grid(row = gridrow, column = 1, sticky="EW")
		gridrow += 1

		gridrow += 1

		self.Chan_Display_label = tk.Label(self.frame023,  text = "Channel: ")
		self.Chan_Display_label.grid(row = gridrow, column = 0, sticky = 'ew')

		self.channels_to_display = ['1']
		self.Chan_Display__choice = ttk.Combobox(self.frame023,values = self.channels_to_display,  width = 18 )
		self.Chan_Display__choice.config(state = "readonly")
		self.Chan_Display__choice.grid(row = gridrow, column = 1, sticky = 'ew')
		self.Chan_Display__choice.set("all")
		gridrow += 1

		self.Line_Display_label = tk.Label(self.frame023,  text = "Line: ")
		self.Line_Display_label.grid(row = gridrow, column = 0, sticky = 'ew')
		
		self.Line_Display__choice = ttk.Combobox(self.frame023,values = ["line 1", "line 2", "all"],  width = 18 )
		self.Line_Display__choice.config(state = "readonly")
		self.Line_Display__choice.grid(row = gridrow, column = 1, sticky = 'ew')
		self.Line_Display__choice.set("line 1")
		gridrow += 1

		self.Scantype_label = tk.Label(self.frame023,  text = "Scan type: ")
		self.Scantype_label.grid(row = gridrow, column = 0, columnspan = 2, sticky = 'w')

		self.Scantype__choice = ttk.Combobox(self.frame023,values = ["1 focus","2 focus"],  width = 18 )
		self.Scantype__choice.config(state = "readonly")
		self.Scantype__choice.grid(row = gridrow, column = 1, sticky = 'ew')
		self.Scantype__choice.set("1 focus")
		gridrow += 1

		
		self.Display_button = tk.Button(self.frame023, text="Display", command=self.Plot_this_file)
		self.Display_button.grid(row = gridrow, column = 0, columnspan =2, sticky="EW")
		gridrow += 1

		self.Transfer_button = tk.Button(self.frame023, text="Transfer curve", command=self.Transfer_extracted)
		self.Transfer_button.grid(row = gridrow, column = 0, columnspan =2, sticky="EW")
		gridrow += 1


		self.figure1 = Figure(figsize=(0.9*win_width/dpi_all,0.85*win_height/dpi_all), dpi = dpi_all)




		gs = self.figure1.add_gridspec(8, 1)


		self.image = self.figure1.add_subplot(gs[0])

		self.image.set_title("sFCS image channel 1")

		self.image.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		
		self.image2 = self.figure1.add_subplot(gs[1])

		self.image2.set_title("sFCS image channel 2")

		self.image2.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		
		self.traces = self.figure1.add_subplot(gs[2:4])
		self.traces.set_title("Traces")
		self.traces.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.traces.set_ylabel('intensity (a.u.)')
		self.traces.set_xlabel('Time (s)')

		self.corr = self.figure1.add_subplot(gs[4:8])

		self.corr.set_title("Correlation curves")
		self.corr.set_ylabel('Diff. Coeff.')
		self.corr.set_ylabel('G (tau)')
		self.corr.set_xlabel('Delay time')

		self.canvas1 = FigureCanvasTkAgg(self.figure1, self.frame04)
		self.canvas1.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)

		self.toolbar = NavigationToolbar2Tk(self.canvas1, self.frame04)
		self.toolbar.update()
		self.canvas1.get_tk_widget().pack()

		self.figure1.tight_layout()

		self.framepb = tk.Frame(frame0)
		self.framepb.pack(side="top", fill="x")


class Sidecut_sFCS:
	def __init__(self,lsm_file_name):
		self.lsm_file_name = lsm_file_name
		#read 1 line scanning FCS files
			#read CZI
		if lsm_file_name.endswith("czi"):
			image = czifile.imread(self.lsm_file_name)
			reshaped_image = image[0, :, :, 0, 0, :, 0]
			# This will give you a new array with the shape (20, 10, 30)
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
		if len(self.array.shape) == 2:
			return self.array
		else:
			return self.array[channel_no-1]
		
	@staticmethod
	def gaussian(x,a,m,s):
		return a*np.exp(-(x-m)**2/(2*s**2))
        
	def isolate_maxima(self, channel_no, bins, lower_lim, upper_lim, t_lower, t_upper):
		self.maxima = []
		self.max_indices = []
		self.bins = []

		if len(self.array.shape) == 3:
			array_to_analyze = self.array[channel_no]
		else:
			array_to_analyze = self.array
		for i, i_array_full in enumerate(array_to_analyze[t_lower:t_upper]):
			if not i % 1000:
				print(i)
			i_array = i_array_full[lower_lim:upper_lim]
			max_value = 0
			max_index = 0
			#calculate membrane pixels with gaussian
			i_array_max = np.max(i_array)
			i_array_std = np.std(i_array)
			n = i if i < 100 else 100
			max_indices_mean = np.mean(self.max_indices[i-n:i])
			try:
				initial_guess = [i_array_max, np.argmax(i_array), i_array_std]
				#print("init ", initial_guess)
				popt, _ = curve_fit(Sidecut_sFCS.gaussian, np.arange(0,len(i_array),1), i_array, p0=initial_guess, maxfev=200)
				#print("popt ",popt)
				max_index = int(popt[1]) #maximum = peak of gaussian
				bins = int(2.5*popt[2]) #bin width = 2.5 sigma
			except (RuntimeError, ValueError):
				#print(i, "fit failed initially, try different starting conditions")
				initial_guess = [i_array_max, max_indices_mean, i_array_std]
				try:
					popt, _ = curve_fit(Sidecut_sFCS.gaussian, np.arange(0,len(i_array),1), i_array, p0=initial_guess, maxfev=200)
					max_index = int(popt[1])
					bins = int(2.5*popt[2])
				except (RuntimeError, ValueError):
					#print(i, "fit failed, using average values")
					if i == 0:
						max_index = (upper_lim - lower_lim) / 2
						bins = (upper_lim - lower_lim) / 2 - 1
						print("round 0 ", max_index, bins)
					else :
						if max_index - bins < 0 or max_index + bins + 1 > len(i_array):
							max_index = max_indices_mean
						if max_index - bins < 0 or max_index + bins + 1 > len(i_array):
							bins = np.mean(self.bins[i-n:i])
			if i == 0:
				if max_index - bins < 0 or max_index + bins + 1 > len(i_array):
					max_index = (upper_lim - lower_lim) / 2
				if max_index - bins < 0 or max_index + bins + 1 > len(i_array):
					bins = (upper_lim - lower_lim) / 2 - 1
			else :
				if max_index - bins < 0 or max_index + bins + 1 > len(i_array):
					max_index = max_indices_mean
				if max_index - bins < 0 or max_index + bins + 1 > len(i_array):
					bins = np.mean(self.bins[i-n:i])
			
			max_index = int(max_index)
			for k in range (-int(bins), int(bins)+1):
				max_value += i_array[max_index + k]
				if i == 0:
					print(max_index+k, i_array[max_index + k])
			"""old code
			max = 0
			for i in range(-bins,bins):
				if np.argmax(i_array) + i < len(i_array) and np.argmax(i_array) + i > 0:
					max += i_array[np.argmax(i_array) + i]
			"""
			self.bins.append(bins)
			self.maxima.append(max_value)
			self.max_indices.append(max_index)
		self.maxima = np.array(self.maxima)
		#self.image.plot(np.arange(0,len(self.max_indices),1), self.max_indices)
		print("maxima array ", channel_no, self.maxima)
		return self.maxima
    
	def maxs_single_autoc_plot(self, channel_no, rep_no, number_of_reps, timestep):
		list_of_reps = np.array_split(self.isolate_maxima(channel_no), number_of_reps)
		y = list_of_reps[rep_no-1]
		time, scorr = corr_py.correlate_full (timestep, y, y)
		plt.xscale("log")
		plt.plot (time, scorr)
		plt.tight_layout()
		plt.show()
        
	def maxs_autoc_plots(self, channel_no, number_of_reps, timestep):
		#plot correlation of each repetition,
		list_of_reps = np.array_split(self.isolate_maxima(channel_no), number_of_reps)
		for i in list_of_reps:
			y = i
			time, scorr = corr_py.correlate_full (timestep, y, y)
			plt.xscale("log")
			plt.plot (time, scorr)
			plt.tight_layout()
			plt.show()
            
		return time, scorr
            
	def maxs_autoc_carpet_plot(self, channel_no, timestep, number_of_reps, plot_title =''):
		list_of_reps = np.array_split(self.isolate_maxima(channel_no), number_of_reps)
		autocorrelation_by_rows = []
		for i in range(len(list_of_reps)):
			y = list_of_reps[i]
			time, scorr = corr_py.correlate_full (timestep, y, y)
			autocorrelation_by_rows.append(scorr)
		fig, ax = plt.subplots(figsize=(100,10))
		im = ax.imshow(autocorrelation_by_rows,origin="lower",cmap='bwr')
		#cbar = ax.figure.colorbar(im, ax=ax,shrink=0.5,location='right', pad =0.003)
		ax.set_title(plot_title)
		plt.show()



######################################## clone of Sidecut_sFCS for 2 focus sFCCS ######################################
class Sidecut_2fsFCS:
	def __init__(self,lsm_file_name):
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
        
	def isolate_maxima(self, channel_no, bins):
		print("self array shape", self.array.shape)
		if len(self.array.shape) == 4:	#dual color
			array_to_analyze = self.array[channel_no]
			self.maxima_2line = np.zeros((self.array.shape[2], self.array.shape[1]), dtype = float)
		else:	#single color
			array_to_analyze = np.array(self.array)
			self.maxima_2line = np.zeros((self.array.shape[1], self.array.shape[0]), dtype = float)
		#print("array to analyze line1", array_to_analyze[0,0,0:100])
		#print("array to analyze line2", array_to_analyze[0,1,0:100])
		for line in range(2):
			self.maxima = []
			#print(line)
			for i in range(array_to_analyze.shape[0]): #time
				max_value = 0
				max_index = 0
				for j in range(0,array_to_analyze.shape[2]):	#y
					if array_to_analyze[i,line,j] < 0:
						print(array_to_analyze[i,line,j])
					if array_to_analyze[i,line,j] > max_value:
						max_value = array_to_analyze[i,line,j]
						max_index = j
				try:
					for j in range (0, bins-1):
						max_value += array_to_analyze[i,line,max_index - j] + array_to_analyze[i,line,max_index + j]

				except:
					print("border pixel")
					max_value = 0
				
				#print(max_value)
				self.maxima_2line[line,i] = max_value
		return self.maxima_2line