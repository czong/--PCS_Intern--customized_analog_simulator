
import os, sys
import decimal
from math import *
from numpy import *
from numpy.fft import fft,fftfreq,fftshift
#from scipy.fftpack import fft, ifft
from scipy.signal import blackmanharris

from traits.api import *
from traitsui.api import *
from chaco.api import Plot, ArrayPlotData
from enable.api import ColorTrait, ComponentEditor
from pdb import set_trace
from numpy.random import randn

from pyface.image_resource import ImageResource
import matplotlib.pyplot as plt

TIME_DOMAIN_CHECK = True

def normal(time):
	return randn(len(time))

def startstopstep(start, stop, step):
	""" like arange but makes sure the array length is a power of 2 for fft's"""
	length = (stop-start)/step
	new_length = 2**( ceil(log2(length)) )
	return arange(start, start+new_length*step, step)

def flicker(time):
	white = normal(time)   #white Gaussian sequence with number of len(time)
	white_fft = fft(white)
	freq = fftfreq(len(time), time[1]-time[0])
	sorted_freq = sorted(abs(freq))
	if (sorted_freq[0]) == 0.0:
		freq[abs(freq).argmin()] = sorted_freq[1]/10.  # we'll take 1/10 to avoid divide by zero
	pink = 1/freq
	pink_fft = pink*white_fft

	# let's estimate total power and make same for pink
	result = ifft(pink_fft)
	white_power = sqrt(sum(abs(white)**2))
	pink_power = sqrt(sum(abs(pink)**2))
	ratio = white_power/pink_power
	return result*ratio

## User Variable Class
class UserVar(HasTraits):
	variable = Str
	value = Float
	slider = Bool(False)
	low = Float
	high = Float

	default_view = View(
		Item('variable', label='Variable Name'),
		Item('value', label='Initial Value'),
		Item('slider', label='Make Slider'),
		Item('low', label='Slider Low'),
		Item('high', label='Slider High'),
		buttons = [OKButton, CancelButton]
	)

	def set_notifier(self, notifier):
		self.on_trait_change(notifier, 'variable')
		self.on_trait_change(notifier, 'value')
		self.on_trait_change(notifier, 'slider')
		self.on_trait_change(notifier, 'low')
		self.on_trait_change(notifier, 'high')

class UserVarUI(HasTraits):
	def default_traits_view(self):
		instance_traits = self._instance_traits()

		all_items = []
		for _ in instance_traits.keys():
			if hasattr(instance_traits[_],'mytype'):
				if instance_traits[_].slider:
					low = instance_traits[_].low
					high = instance_traits[_].high
					all_items.append(Item(_, editor = RangeEditor(low=low, high=high, format="%4.2g")))
				else:
					all_items.append(Item(_))

		traits_view = View( \
						*tuple(all_items)
		        )
		return traits_view

	def clone(self):
		result = UserVarUI() # new copy	
		instance_traits = self._instance_traits()
		for key in instance_traits.keys():
			if hasattr(instance_traits[key], 'mytype'):
				value = getattr(self, key)
				low = instance_traits[key].low
				high = instance_traits[key].high
				slider = instance_traits[key].slider
				result.add_trait(key, Float(value, mytype=True, low=low, high=high, slider=slider ))
		return result


# table editor for UserVar class
_the_cols = [ 'variable', 'value', 'slider', 'low', 'high' ]
table_columns = [ ObjectColumn( name = _, label = _ , editable = True, horizontal_alignment = 'center') for _ in _the_cols]
table_editor = TableEditor(
	columns = table_columns,
	auto_size          = True,
	auto_add          = True,
	show_toolbar       = True,
	selected           = 'selected',
	selection_color    = 0x000000,
	selection_bg_color = 0xFBD391,
	row_factory = UserVar,
	deletable = True,
)


## MAIN CLASS
class FFTutor(HasTraits):
        precision = 3
        ########################################################
        ####################### ADC parameters
        Fs_ADC = Expression(1.9e9)
        Fin_ADC = Expression(22e6)
        amp_ADC = Float(0.707)
        c1_ADC = Expression(.5e-12)
        c2_ADC = Expression(3e-12)
        c3_ADC = Expression(3e-12)
        
        b1_ADC = Float(2.)
        b2_ADC = Float(2.)
        b3_ADC = Float(2.2)                
        bz_ADC = Float(1.)  
        
        Signal_BW_ADC = Expression(1e6)      
        Integ_string = String('Integ:')
        Integ_low_ADC = Expression(.2e6)
        Integ_high_ADC = Expression(40e6)
        
        fft_string = String('FFT:')
        fft_low_index_ADC = Expression(.1e6)
        fft_high_index_ADC = Expression(60e6)
        
        # constants:
        gm = 666e-6
        LF = .5         
        N = int(1e5)
        
        snrnew_ADC = String
        snr_ADC = Float
        sigdb_ADC = Float
        nt_ADC = Float
        
        update_ADC = Button('Update')    
        
        
        ########################################################
        ####################### UI of Tab for signal from file
        # Input for signal from file	
	Fsample = Expression(160e6, desc='Sampling Frequency')
	Finput = Expression(10e6, desc='Input or Fundamental Tone')
	Kaiser_beta_file = Float(9.6, desc='Parameter for Kaiser Window')
	y_offset = Float(2048.)
	Signal_BW = Expression(2e6,desc='The bandwidth of signal to calculate SNR')
	Integ = String
	Integ_low = Expression(0,desc='Low Frequency for noise power integration')
	Integ_high = Expression(0,desc='High Frequency for noise power integration')
	AVG = Int(114,desc='Specify how many FFT blocks')
	Data_File = File
	Full_Scale = Expression('2**12', desc='Full Scale Input - can be an expression, e.g.  2**12')	
	# Other for signal from file
	active_plot_file = Button('Reload')
	time_domain = String
	#time_yMax = Float(0.0)
	#time_yMin = Float(0.0)
	#time_yAverage = Float(0.0)
	#time_peakToPeak = Float(0.0)x

        time_yMax = Expression(0.0)
	time_yMin = Expression(0.0)
	time_yAverage = Expression(0.0)
	time_peakToPeak = Expression(0.0)
	
	f_domain = String
	#f_min = Float(0.0)
	#f_max = Float(0.0)
	f_min = Expression(0)
	f_max = Expression(0)
	
	nt = Float(0.0)
	sigdb = Float(0.0)
	snrnew = Expression(0.0)
	snr = Float(0.0)
	data_begin_lines = Float(10)
	line_br = Str	
	# Index and value range for FFT and time domain plots for signal from file
	fft_plot_string = String
	fft_plot_xlow_file=Expression(0,desc='lower limit of signal frequency')
	fft_plot_xhigh_file=Expression(0,desc='upper limit of signal frequency')
	fft_plot_ylow_file=Expression(0,desc='lower limit of signal power')
	fft_plot_yhigh_file=Expression(0,desc='upper limit of signal frequency')
        time_plot_string = String
	time_plot_xlow_file=Expression(0,desc='lower limit of signal index in time domain (in cycles)')
	time_plot_xhigh_file=Expression(10,desc='upper limit of signal index in time domain (in cycles)')
	time_plot_ylow_file=Expression(0,desc='lower limit of signal value in time domain')
	time_plot_yhigh_file=Expression(0,desc='upper limit of signal value in time domain')	
	# Property depending on the input from the file
	plot = Instance(Plot)
	data_file = ndarray(1)
	time_file = ndarray(1)

		
	#########################################################
	#######################  UI of Tab for signal from user
	# MISC
	status = Str('')
	logostr = Str
	add_var = Button('+')
	active_plot_user = Button('Active')
	# USER VARS
	the_variables  = List( UserVar, value=[] )
	the_variables_ui = Instance(UserVarUI)
	# INPUTS	
	time_user = Expression('startstopstep(0.0, ncycles/f0, 1/fsample)', desc='time expression')
	signal = Expression('sin(2*pi*f0*time_user)+anoise*white(time_user)', desc='signal expression')
	signal_frequency = Expression('f0', desc='signal of frequency for S/N calculation')
	signal_bw = Expression('0.02*f0', desc='signal bandwidth for S/N calculation')
	kaiser_beta_user = Float('9.0')
	# Dependent properties
	signal_evaled = Array(dtype=dtype('float64'))
	time_evaled = Array(dtype=dtype('float64'))
	time_evaled_non_win = Array(dtype=dtype('float64'))
	frequency = Array(dtype=dtype('float64'))
	signal_fft = Array(dtype=dtype('float64'))
	snr = Float('0.0')	
	# Index and value range for FFT and time domain plots for signal from user
	fft_plot_xlow_user = Float(0.0)
	fft_plot_xhigh_user = Float(40.0)
	fft_plot_ylow_user = Float(-120.)
	fft_plot_yhigh_user = Float(0.0)
	time_plot_xlow_user = Float(0.0)
	time_plot_xhigh_user = Float(2.0)
	time_plot_ylow_user = Float(-4.0)
	time_plot_yhigh_user = Float(4.0)

	#### Watchdogs
	#### The purpose of a watchdog is to force traits notifications.  An oddity of traits notifications
	#### is that notifications only seem to propagate to the penultimate level.  For example if
	#### signal_watchdog were not defined, then signal_fft would not be updated whenever one of its antecedents
	#### was triggered.  But since signal_watchdog depends on signal_fft traits forces a refresh of signal_fft
	#### because signal_watchdog depends on it.  There is no other purpose for the watchdogs except to
	#### force evaluation of needed top level properties
	#signal_watchdog = Property(depends_on='signal_fft')
	#frequency_watchdog = Property(depends_on='frequency')

	
        ##################################### Plot instance for both FFT and Time domain plot
	fft_plot = Instance(Plot)
	time_plot = Instance(Plot)


	#################################### view design	
	devfault_view = View(
	   
		VSplit(
			Tabbed(
                                  HGroup(   
                                     VGroup(                                           
		                          Item('Fsample',label='Fs'),
		                          Item('Finput'),
		                          Item('AVG'),
		                          Item('Kaiser_beta_file',label='Kaiser Beta'),
		                          Item('y_offset',label='DC offset'),
		                          Item('Full_Scale'),
		                          Item('Signal_BW'),
		                          HGroup(
		                              Item('Integ',label='Power Integration:',style='readonly'),
		                              Item('Integ_low',label='low f'),
		                              Item('Integ_high',label='high f'),		                          
		                          ),
		                          
		                          Item('line_br', style = 'readonly', show_label=False),
		                          HGroup(
		                                  Item('time_domain', style = 'readonly',label='Amplitude of Time Domain:'),
		                                  Item('time_yMin', style = 'readonly',label='Min:'),
		                                  Item('time_yMax', style = 'readonly',label='Max:'),
		                                  Item('time_yAverage', style = 'readonly',label='Average:'),
		                                  Item('time_peakToPeak', style = 'readonly',label='Peak to Peak:')
		                          ),
		                          HGroup(	
		                                  Item('f_domain',style = 'readonly',label='Range of Frequency Domain:'),	                        
		                                  Item('f_min', style = 'readonly',label='Min:'),
		                                  Item('f_max', style = 'readonly',label='Max:')
		                          ),
		                          
		                          Item('line_br', style = 'readonly', show_label=False),
		                          
		                          HGroup(
		                                  Item('fft_plot_string',label='FFT Plot', style = 'readonly'),
		                                  Item('fft_plot_xlow_file',label='X min'),
		                                  Item('fft_plot_xhigh_file',label='X max'),
		                                  Item('fft_plot_ylow_file',label='Y min'),
		                                  Item('fft_plot_yhigh_file',label='Y max')
		                          ),
		                          HGroup(
		                                  Item('time_plot_string',label='Time Plot', style = 'readonly'),
		                                  Item('time_plot_xlow_file',label='X min'),
		                                  Item('time_plot_xhigh_file',label='X max'),
		                                  Item('time_plot_ylow_file',label='Y min'),
		                                  Item('time_plot_yhigh_file',label='Y max')
		                          ),
                                          Item('line_br', style = 'readonly', show_label=False),
                                          HGroup(
		                                  
		                                  Item('Data_File'),
		                                  Item('active_plot_file', show_label=False),
		                          ),
		                          
		                          HGroup(
		                              Item('nt', style = 'readonly'),
		                              Item('sigdb', style = 'readonly'),
		                              Item('snrnew', style = 'readonly'),
		                              
		                          ),
		                          Item('data_begin_lines', style = 'readonly',label='Data begins at line:'),
		                           		                           
                                     ),
                                     label='File' 
			         ),
			             
			             HGroup(
				
				            VGroup(
					           Item('the_variables_ui', show_label=False, style='custom'),
					           #spring,
                                                   Item('line_br', style = 'readonly', show_label=False),
					           Item('time_user'),
					           Item('signal'),
					           HGroup(
						          Item('signal_bw'),
						          Item('signal_frequency'),
						          Item('kaiser_beta_user')
					           ),
					           HGroup(
						          Item('fft_plot_xlow_user'),
						          Item('fft_plot_xhigh_user'),
						          Item('fft_plot_ylow_user'),
						          Item('fft_plot_yhigh_user'),
					           ),
					           HGroup(
						          Item('time_plot_xlow_user'),
						          Item('time_plot_xhigh_user'),
						          Item('time_plot_ylow_user'),
						          Item('time_plot_yhigh_user'),
					           ),	
					           HGroup(
					                  Item('snr', style='readonly'),
#					                  Item('status', show_label=True),
					                  show_border = True
				                   ),
					           Item('active_plot_user',show_label=False),
					           Item('line_br', style = 'readonly', show_label=False),
					           #spring,
					           show_border = True
				            ),
				            VGroup(
					           HGroup( 
						          Item('add_var', show_label=False),
						          spring,
						          Item('logostr', show_label=False, editor=ImageEditor(image=ImageResource('logo', search_path=['.']))),
					           ),
					           Item( 'the_variables',label = 'Results',id='the_variables',editor = table_editor,show_label = False,
					           ),
				            ),
				            show_border = True,
				            label='User'
			             ),
			             
			         VGroup(
			             HGroup(
			                 VGroup(
			                     Item('Fs_ADC',label='Fs'),
			                     Item('Fin_ADC',label='Fin'),
			                     Item('amp_ADC',label='Amp'),
			                     Item('line_br', style = 'readonly', show_label=False),
			                     VGroup(
			                         Item('Integ_string',show_label=False,style='readonly'),
			                         HGroup(			                     
			                             Item('Integ_low_ADC',label='low:'),
			                             Item('Integ_high_ADC',label='high:'),
                                                 ),
                                             
                                                 Item('fft_string',show_label=False,style='readonly'),
                                                 HGroup(                                             
			                             Item('fft_low_index_ADC',label='low'),
			                             Item('fft_high_index_ADC',label='high'),			                 
			                         ),
			                         Item('Signal_BW_ADC',label='Sig BW'),
			                     ),
			                 ),
			                 VGroup(
			                     VGroup(
			                         Item('c1_ADC',label='c1'),
			                         Item('c2_ADC',label='c2'),
			                         Item('c3_ADC',label='c3')
			                     ),
			                     VGroup(
			                         Item('b1_ADC',label='b1'),
			                         Item('b2_ADC',label='b2'),
			                         Item('b3_ADC',label='b3'),
			                         Item('bz_ADC',label='bz'),			                     
			                     )
			                 ),
			                     
			             ),
			             Item('line_br', style = 'readonly', show_label=False),
			             HGroup(
			                     Item('snrnew_ADC',label='SNR',style = 'readonly'),
			                     Item('sigdb_ADC',label='signal power',style = 'readonly'),			                     
			                     Item('nt_ADC',label='noise power',style = 'readonly'),
			                     Item('update_ADC',show_label=False),			                     			         
			             ),
			         label='ADC',
			         springy=False
			         ),
			         
			),
			VGroup(
				HGroup(
					Item('fft_plot', editor=ComponentEditor(), show_label=False,springy=True),
					Item('time_plot', editor=ComponentEditor(), show_label=False,springy=True),   # time domain plot for the signal
					show_border = True
				),
				
				show_border = True,
				springy = True
			)
		),
		#width=1200, height=800, 
		resizable=True, 
		#buttons = ["OK"],
		#icon=ImageResource(name='palma_icon.png', search_path=[os.path.curdir]),
		title="PCS Spectrum Analyzer"
	   
	)

	def __init__(self, *args, **kw):
		super(FFTutor, self).__init__(*args, **kw)
		#self.time_plot_xlow_file = 1/self.Finput
		#self.time_plot_xhigh_file = 10/self.Finput
		
		# change all the value to engineering format with pre-defined precision
		self.updateEngFormat();
		
		x = [1.0, 2.0]
		y = [1.0, 2.0]
		t = [1.0, 2.0]
		amp = [1.0, 2.0]

		self.the_variables_ui = UserVarUI()
		uv = UserVar(variable='f0', value=10.0, slider=True, low=1, high=40.)
		uv.set_notifier(self.on_user_var_changed)
		self.the_variables.append(uv)
		self.load_the_variables_ui()
		uv = UserVar(variable='anoise', value=0.0, slider=True, low=0.0, high=0.5)
		uv.set_notifier(self.on_user_var_changed)
		uv.set_notifier(self.on_user_var_changed)
		self.the_variables.append(uv)
		self.load_the_variables_ui()
		uv = UserVar(variable='fsample', value=80., slider=True, low=1, high=160.)
		uv.set_notifier(self.on_user_var_changed)
		self.the_variables.append(uv)
		self.load_the_variables_ui()
		uv = UserVar(variable='ncycles', value=100., slider=True, low=1, high=1000.)
		uv.set_notifier(self.on_user_var_changed)
		self.the_variables.append(uv)
		self.load_the_variables_ui()
		
		###################################################################
                ################################ detect trait change for user input
                ## update the signal from input
		self.on_trait_change(self.updateFromUser, 'time_user')
		self.on_trait_change(self.updateFromUser, 'signal')
		self.on_trait_change(self.updateFromUser, 'signal_frequency')
		self.on_trait_change(self.updateFromUser, 'signal_bw')
		self.on_trait_change(self.updateFromUser, 'kaiser_beta_user')	
		# update the range of the plot for the signal from input
		self.on_trait_change(self.time_plot_change_xlow_user,'time_plot_xlow_user')
		self.on_trait_change(self.time_plot_change_xhigh_user,'time_plot_xhigh_user')
		self.on_trait_change(self.time_plot_change_ylow_user,'time_plot_ylow_user')
		self.on_trait_change(self.time_plot_change_yhigh_user,'time_plot_yhigh_user')
		self.on_trait_change(self.fft_plot_change_xlow_user,'fft_plot_xlow_user')
		self.on_trait_change(self.fft_plot_change_xhigh_user,'fft_plot_xhigh_user')		
		self.on_trait_change(self.fft_plot_change_ylow_user,'fft_plot_ylow_user')
		self.on_trait_change(self.fft_plot_change_yhigh_user,'fft_plot_yhigh_user')
		self.on_trait_change(self.active_user,'active_plot_user')		
		
		###################################################################
		################################ detect trait change for file input
		# update the signal from file						
		self.on_trait_change(self.updateFromFile,'Fsample')
	        self.on_trait_change(self.updateFromFile,'Finput')
	        self.on_trait_change(self.updateFromFile,'AVG')
	        self.on_trait_change(self.updateFromFile,'Kaiser_beta_file')
	        self.on_trait_change(self.updateFromFile,'y_offset')
	        self.on_trait_change(self.updateFromFile,'Full_Scale')
	        self.on_trait_change(self.updateFromFile,'Signal_BW')
		self.on_trait_change(self.updateFromFile, 'Integ_low')
		self.on_trait_change(self.updateFromFile, 'Integ_high')		        
	        self.on_trait_change(self.read_data,'Data_File')		
		# update the range of the plot for the signal from file
		self.on_trait_change(self.fft_plot_change_xlow_file,'fft_plot_xlow_file')
		self.on_trait_change(self.fft_plot_change_xhigh_file,'fft_plot_xhigh_file')
		self.on_trait_change(self.fft_plot_change_ylow_file,'fft_plot_ylow_file')
		self.on_trait_change(self.fft_plot_change_yhigh_file,'fft_plot_yhigh_file')						
		self.on_trait_change(self.time_plot_change_xlow_file,'time_plot_xlow_file')
		self.on_trait_change(self.time_plot_change_xhigh_file,'time_plot_xhigh_file')
		self.on_trait_change(self.time_plot_change_ylow_file,'time_plot_ylow_file')
		self.on_trait_change(self.time_plot_change_yhigh_file,'time_plot_yhigh_file')
		self.on_trait_change(self.active_file,'active_plot_file')

		################################ detect trait change for ADC param
		''''
		self.nz1 = self.gm/(float(self.c1_ADC)*float(self.Fs_ADC))
                self.nz2 = self.gm/(float(self.c2_ADC)*float(self.Fs_ADC))
                self.nz3 = self.gm/(float(self.c3_ADC)*float(self.Fs_ADC)) 
		'''
		'''
		self.on_trait_change(self.update_ADC,'Fs_ADC')
		self.on_trait_change(self.update_ADC,'Fin_ADC')		
		self.on_trait_change(self.update_ADC,'c1_ADC')
		self.on_trait_change(self.update_ADC,'c2_ADC')						
		self.on_trait_change(self.update_ADC,'c3_ADC')
		self.on_trait_change(self.update_ADC,'b1_ADC')				
		self.on_trait_change(self.update_ADC,'b2_ADC')				
		self.on_trait_change(self.update_ADC,'b3_ADC')
		self.on_trait_change(self.update_ADC,'bz_ADC')														
		self.on_trait_change(self.update_ADC,'Signal_BW_ADC')
		'''
		self.on_trait_change(self.updateADCmethod,'update_ADC')
		
			
							
		# initialize ArrayPlotData		
		self.plot_data = ArrayPlotData(x=x, y=y,t=t,amp = amp)
		
	########################################################################
	#############################   update the signal from ADC file
	def plot_fft_time(self,plot_data):
	    self.fft_plot = Plot(plot_data)
	    self.time_plot = Plot(plot_data)
	    self.time_plot.index_axis.tick_label_formatter = self.sci_formatter
	    self.fft_plot.index_axis.tick_label_formatter = self.sci_formatter
	    self.renderer = self.fft_plot.plot(("x", "y"), type='line', color='magenta')	       	  
	    self.fft_plot.index_axis.title = 'frequency [Hz]'
	    self.fft_plot.value_axis.title = 'amplitude [dBc]'
	    self.fft_plot.title="FFT"
            self.renderer = self.time_plot.plot(("t","amp"),type='line',color='magenta')	
	    self.time_plot.index_axis.title = 'time [s]'
	    self.time_plot.value_axis.title = 'amplitude []'
	    self.time_plot.title = "Time"
	
	def set_fft_plot_range(self,index_low,index_high,value_low=0,value_high=0):
	    self.fft_plot.index_range.low = index_low
            self.fft_plot.index_range.high = index_high
	    if value_low!=0 and value_high!=0:
	        self.fft_plot.value_range.low = value_low
		self.fft_plot.value_range.high = value_high		

	def set_time_plot_range(self,index_low,index_high,value_low=0,value_high=0):
	    self.time_plot.index_range.low = index_low
            self.time_plot.index_range.high = index_high
	    if value_low!=0 and value_high!=0:
	        self.time_plot.value_range.low = value_low
		self.time_plot.value_range.high = value_high		
			
	def updateADCmethod(self):
	    self.time_ADC,self.data_ADC,self.out, self.CMP = self.redigitize_ADC()

	    self.block_size = self.calc_block_size(self.data_ADC,1)
	    self._full_scale = max(self.data_ADC)-min(self.data_ADC)
	    self.f = self.calc_f(self.data_ADC,self.block_size,self.Fs_ADC,self._full_scale)	        	
	    self.Yf,self.Yf_linear = self.calc_Yf(self.data_ADC,0,self._full_scale,self.block_size,1,self.Kaiser_beta_file)
	    self.snrnew_ADC, self.snr_ADC, self.sigdb_ADC, self.nt_ADC = self.calc_snrnew(self.data_ADC,self.Yf_linear,self.Fin_ADC,self.Signal_BW_ADC,self.f,self.Integ_low_ADC,self.Integ_high_ADC,2*self.amp_ADC,2*self.amp_ADC)
	    
	    self.plot_data.set_data('x', self.f)
            self.plot_data.set_data('y', self.Yf)
	    self.plot_data.set_data('t', self.time_ADC)
	    self.plot_data.set_data('amp',self.data_ADC)
	    self.plot_fft_time(self.plot_data)
	    self.set_fft_plot_range(self.fft_low_index_ADC,self.fft_high_index_ADC)
	    self.set_time_plot_range(self.time_ADC[1],10/float(self.Fin_ADC))	  		
		
	def redigitize_ADC(self):
	    N = self.N
	    t = arange(0,N)/float(self.Fs_ADC)
	    In = self.amp_ADC*sin(2*pi*t*float(self.Fin_ADC))
	    
	    self.nz1 = self.gm/(float(self.c1_ADC)*float(self.Fs_ADC))
            self.nz2 = self.gm/(float(self.c2_ADC)*float(self.Fs_ADC))
            self.nz3 = self.gm/(float(self.c3_ADC)*float(self.Fs_ADC))
	 
	    I0 = zeros(N)
	    I1 = zeros(N)
	    I2 = zeros(N)
	    I3 = zeros(N)	    	    
	    CMP = zeros(N)
	    out = zeros(N)
	    Dout = zeros(N)
	    
	    for i in range(1,N):
	        I0[i]=In[i]-CMP[i-1]
	        I1[i]=self.nz1*I0[i-1]+I1[i-1]
	        I2[i]=self.nz2*(I1[i-1]-self.bz_ADC*I3[i-1])+I2[i-1]
	        I3[i]=self.nz3*I2[i-1]+I3[i-1]
	        
	        out[i]=self.b3_ADC*I3[i]+self.b2_ADC*I2[i]+self.b1_ADC*I1[i]-self.LF*CMP[i-1]  	    	    
                
                if out[i]<=-14./15:
                    CMP[i]=-1
                    Dout[i]=0
                elif out[i]<=-12./15:
                    CMP[i]=-13./15
                    Dout[i]=1
                elif out[i]<=-10./15:
                    CMP[i]=-11./15
                    Dout[i]=2
                elif out[i]<=-8./15:
                    CMP[i]=-9./15
                    Dout[i]=3
                elif out[i]<=-6./15:
                    CMP[i]=-7./15
                    Dout[i]=4
                elif out[i]<=-4./15:
                    CMP[i]=-5./15
                    Dout[i]=5
                elif out[i]<=-2./15:
                    CMP[i]=-3./15
                    Dout[i]=6
                elif out[i]<=0:
                    CMP[i]=-1./15
                    Dout[i]=7
                elif out[i]<=2./15:
                    CMP[i]=1./15
                    Dout[i]=8
                elif out[i]<=4./15:
                    CMP[i]=3./15
                    Dout[i]=9
                elif out[i]<=6./15:
                    CMP[i]=5./15
                    Dout[i]=10
                elif out[i]<=8./15:
                    CMP[i]=7./15
                    Dout[i]=11
                elif out[i]<=10./15:
                    CMP[i]=9./15
                    Dout[i]=12
                elif out[i]<=12./15:
                    CMP[i]=11./15
                    Dout[i]=13
                elif out[i]<=14./15:
                    CMP[i]=13./15
                    Dout[i]=14    
	        else:
                    CMP[i]=1
                    Dout[i]=15
	    
	    Dout-=7.5
	    return [t,Dout,out,CMP]    	
		
													
	########################################################################
	#############################  update the signal from file	
	    					
	def updateFromFile(self):
	        
	        self.time_yMin = str(min(self.data_file))
		self.time_yMax = str(max(self.data_file))
		self.time_yAverage = str((float(self.time_yMax)+float(self.time_yMin))/2.0)
		self.time_peakToPeak = str(float(self.time_yMax)-float(self.time_yMin))

		self.time_yMax = self.toEngFormat(self.time_yMax)
	        self.time_yMin = self.toEngFormat(self.time_yMin)
	        self.time_yAverage = self.toEngFormat(self.time_yAverage)
	        self.time_peakToPeak = self.toEngFormat(self.time_peakToPeak)

	        
	        
	        
	        self.recalculate_file()	        
                self.plot_data.set_data('x', self.f)
                self.plot_data.set_data('y', self.Yf)
		self.plot_data.set_data('t', self.time_file)
		self.plot_data.set_data('amp',self.data_file)
		
		if self.AVG == 0:
		  self.fft_plot.index_range.low = min(self.f)
		  self.fft_plot.index_range.high = max(self.f)
		else:
		  self.fft_plot.index_range.low = min(abs(self.f))
		  self.fft_plot.index_range.high = max(abs(self.f))
		  				    
		self.fft_plot.value_range.low = min(self.Yf)
		self.fft_plot.value_range.high = max(self.Yf)		
		self.time_plot.index_range.low = float(self.time_plot_xlow_file)/float(self.Finput)
		self.time_plot.index_range.high = float(self.time_plot_xhigh_file)/float(self.Finput)
		self.time_plot.value_range.low = min(self.data_file)
		self.time_plot.value_range.high = max(self.data_file)	
		
		# update the user input too
		self.fft_plot_xlow_file = self.toEngFormat(min(abs(self.f)))
		self.fft_plot_xhigh_file = self.toEngFormat(max(abs(self.f)))
		self.fft_plot_ylow_file = self.toEngFormat(min(self.Yf))
		self.fft_plot_yhigh_file = self.toEngFormat(max(self.Yf[ceil(len(self.Yf)/2)+8:]))		
		#self.time_plot_xlow_file = 1/self.Finput
		#self.time_plot_xhigh_file = 10/self.Finput
		self.time_plot_ylow_file = self.toEngFormat(min(self.data_file))
		self.time_plot_yhigh_file = self.toEngFormat(max(self.data_file))
		
		self.f_min = str(min(self.f))
		self.f_max = str(max(self.f))		
	        self.f_min = self.toEngFormat(self.f_min)
	        self.f_max = self.toEngFormat(self.f_max)
	       
	        		
		# update these three expression variables to engineering format
		#self.Fsample = self.toEngFormat(self.Fsample)
	        #self.Finput = self.toEngFormat(self.Finput)
	        #self.Signal_BW = self.toEngFormat(self.Signal_BW)
		
		#self.calc_nt()
		#self.calc_sigdb()
		#self.calc_snrnew() 
		      

	def time_plot_change_xlow_file(self):	 
            #self.time_plot_xlow_file = self.toEngFormat(self.time_plot_xlow_file)	    
	    self.time_plot.index_range.low = float(self.time_plot_xlow_file)/float(self.Finput)
	    self.time_plot.request_redraw()
	def time_plot_change_xhigh_file(self):
	    #self.time_plot_xhigh_file = self.toEngFormat(self.time_plot_xhigh_file)
	    self.time_plot.index_range.high = float(self.time_plot_xhigh_file)/float(self.Finput)	    
	    self.time_plot.request_redraw()        
        def time_plot_change_ylow_file(self):
            #self.time_plot_ylow_file = self.toEngFormat(self.time_plot_ylow_file)
            self.time_plot.value_range.low = float(self.time_plot_ylow_file)
	    self.time_plot.request_redraw()	
	def time_plot_change_yhigh_file(self):
	    #self.time_plot_yhigh_file = self.toEngFormat(self.time_plot_yhigh_file)
	    self.time_plot.value_range.high = float(self.time_plot_yhigh_file)	
	    self.time_plot.request_redraw()	
	
	def fft_plot_change_xlow_file(self):
	    #self.fft_plot_xlow_file = self.toEngFormat(self.fft_plot_xlow_file)
	    self.fft_plot.index_range.low = float(self.fft_plot_xlow_file)
	    self.fft_plot.request_redraw()
	def fft_plot_change_xhigh_file(self):
	    #self.fft_plot_xhigh_file = self.toEngFormat(self.fft_plot_xhigh_file)
	    self.fft_plot.index_range.high = float(self.fft_plot_xhigh_file)	    
	    self.fft_plot.request_redraw()        
        def fft_plot_change_ylow_file(self):
	    #self.fft_plot_ylow_file = self.toEngFormat(self.fft_plot_ylow_file)            
            self.fft_plot.value_range.low = float(self.fft_plot_ylow_file)
	    self.fft_plot.request_redraw()	
	def fft_plot_change_yhigh_file(self):
	    #self.fft_plot_yhigh_file = self.toEngFormat(self.fft_plot_yhigh_file)	    
	    self.fft_plot.value_range.high = float(self.fft_plot_yhigh_file)	
	    self.fft_plot.request_redraw()			
	
	def active_file(self):
	    self.updateFromFile()
	    self.time_plot_change_xlow_file()
	    self.time_plot_change_xhigh_file()
	    self.time_plot_change_ylow_file()
	    self.time_plot_change_yhigh_file()
	    self.fft_plot_change_xlow_file()
	    self.fft_plot_change_xhigh_file()
	    self.fft_plot_change_ylow_file()
	    self.fft_plot_change_yhigh_file()
	    self.time_plot.index_axis.tick_label_formatter = self.sci_formatter
	    self.fft_plot.index_axis.tick_label_formatter = self.sci_formatter
	
	def read_data(self):
		try:
			data_file = open(self.Data_File).read()
		except:
			print 'no such file', self.Data_File
			self.data_file = zeros(1)
			self.time_file = zeros(1)			
			return

		content = data_file.split('\n')
		if 0:
			data_list = []
			time_list = []
			
			for line in content:
				if not line: continue
				else:
					fields = line.split(',')
					if fields[0]=='\r':
						continue
					else:	
						data_list.append(float(fields[0]))
						time_field = fields[1].strip()
						multiplier = 1.
						if time_field.endswith('ms'):
							time_field = time_field[:-2]
							multiplier = 1E-3
						elif time_field.endswith('us'):
							time_field = time_field[:-2]
							multiplier = 1E-6
						elif time_field.endswith('ns'):
							time_field = time_field[:-2]
							multiplier = 1E-9
						elif time_field.endswith('ps'):
							time_field = time_field[:-2]
							multiplier = 1E-12
						elif time_field.endswith('s'):
							time_field = time_field[:-1]
							multiplier = 1.
						time_field = float(time_field)*multiplier
						time_list.append(time_field)

		multipliers = {'ms':1E-3, 'us':1E-6, 'ns': 1E-9, 'ps':1E-12, 'fs':1E-12, 's':1.}
		content = [_.strip() for _ in content]
		content = [_ for _ in content if _] # remove blank lines
		content = [_.split(',') for _ in content]
		data_list = [float(_[0]) for _ in content]
		#time_list = [0]
		time_list = [_[1].split() for _ in content ]
		time_list = [float(_[0])*multipliers[_[1]] if len(_)>1 else float(_[0]) for _ in time_list]
								
		self.data_file = asarray(data_list) # shadow the data
		self.time_file = asarray(time_list) # shadow the time
		self.fft_plot = Plot(self.plot_data)
		self.time_plot = Plot(self.plot_data)
		self.time_plot.index_axis.tick_label_formatter = self.sci_formatter
		self.fft_plot.index_axis.tick_label_formatter = self.sci_formatter
		
		
		
		##### update of Fsample and Finput depending on the file we read from
		self.Fsample = self.toEngFormat(str(1/(self.time_file[1]-self.time_file[0])))
		Yf_positive_f = self.Yf[self.f>0]
		peaks_index = self.find_peaks(Yf_positive_f)
		peaks_index_1 = argmax(Yf_positive_f[peaks_index])
		self.Finput = self.toEngFormat(str(self.f[self.f>0][peaks_index][peaks_index_1]))
		
		self.Integ_low = str(self.f_signal_low)
		self.Integ_high = str(self.f_signal_high)
		self.Integ_low = self.toEngFormat(self.Integ_low)
	        self.Integ_high = self.toEngFormat(self.Integ_high)	
		
		#self.updateFromFile()
	   	        
	        self.renderer = self.fft_plot.plot(("x", "y"), type='line', color='magenta')	       	  
	        self.fft_plot.index_axis.title = 'frequency [Hz]'
	        self.fft_plot.value_axis.title = 'amplitude [dBc]'
	        self.fft_plot.title="FFT"
	        
	        self.renderer = self.time_plot.plot(("t","amp"),type='line',color='magenta')	
	        self.time_plot.index_axis.title = 'time [s]'
	        self.time_plot.value_axis.title = 'amplitude []'
	        self.time_plot.title = "Time"
	    		       
	
			
	def recalculate_file(self):
	        self.block_size = self.calc_block_size(self.data_file,self.AVG)
	        self._full_scale = self.calc_full_scale()
	        self.f = self.calc_f(self.data_file,self.block_size,self.Fsample,self._full_scale)	        	
	        self.Yf,self.Yf_linear = self.calc_Yf(self.data_file,self.y_offset,self._full_scale,self.block_size,self.AVG,self.Kaiser_beta_file)
	        '''
	        self.Yp = self.calc_Yp()
		self.calc_nt(self.data_file,self.Fsample,self.block_size,self.Signal_BW,self.Finput,self.AVG,)
		self.calc_sigdb()
		'''
		self.snrnew, self.snr, self.sigdb, self.nt = self.calc_snrnew(self.data_file,self.Yf_linear,self.Finput,self.Signal_BW,self.f,self.Integ_low,self.Integ_high,self._full_scale,self.time_peakToPeak)

	def calc_block_size(self,data,block_num):
		total_length = len(data)	
		block_size = float(total_length)/float(block_num)
		block_size = 2**floor(log2(block_size))    # to ensure the length of each block is 2**m
		return block_size      # return the length of each block in time domain, how many points, basically

	def calc_f(self,data,block_size,Fsample,full_scale):
		if len(data) == 1: # no data file
			return arange(-10., 10., 1.)
		f_num = arange(-block_size/2.,block_size/2.,1)
		f = f_num*float(Fsample)/block_size      # it returns the frequencies ranging from -Fsample/2 to Fsample/2
		return f

        def calc_full_scale(self):
		return eval(self.Full_Scale)	
		
	def calc_Yf(self,data,offset,full_scale,block_size,block_num,Kaiser_beta):
		if len(data) == 1: # no data file
			return -(arange(-10., 10., 1.))**2		
		y = data	
                B = y-offset
		B /= (full_scale/2)            				
		win = kaiser(block_size,Kaiser_beta)    # windowing
		Y = zeros((block_num,block_size))
		for i in range(int(block_num)):
			B1 = B[((i)*block_size):(((i+1)*block_size))]
			Yf = B1*win
			Y[i,:] = 4*abs(fftshift(fft(Yf)))
		Y = sum(Y, axis=0)/block_num
		Y /= block_size
		#self._yf_linear = Y
		Y_log = 20*log10(Y)	
		return [Y_log,Y]

        '''
	def calc_Yp(self):
		if len(self.data_file) == 1: # no data file
			return -(arange(-10., 10., 1.))**2		
		y = self.data_file
		B = y-self.y_offset
		win = kaiser(self.block_size,self.Kaiser_beta_file)
		Y = zeros((self.AVG,self.block_size))
		B1 = B[0:self.block_size]
		Yf = B1*win
		Yf=-20*log10(.25*self.block_size)+20*log10(abs(fftshift(fft(Yf))))-20*log10(self.y_offset)		
		Y[0,:] = 10.**(Yf/10.)		
		for i in range(1,int(self.AVG)):
			B1 = B[((i)*self.block_size):(((i+1)*self.block_size))]
			Yf = B1*win
			Yf = -20*log10(.25*self.block_size)+20*log10(abs(fftshift(fft(Yf))))-20*log10(self.y_offset)
			Y[i,:] = 10.**(Yf/10.)
			Y[i,:] = (Y[i,:]+Y[(i-1),:])/2		
		Yp = Y[self.AVG-1,:]      # only return the last one? Why?
		Yf = 10.*log10(Yp)	
		return Yp
        '''
        
        '''
   	def calc_nt(self,data,Fsample,block_size,Signal_BW,Finput,block_num):
		#****************** Post Process Data to get SNR ******************
		if len(data) == 1: # no data file
			return -(arange(-10., 10., 1.))**2		
		y = data		
		Fbw_h=float(Fsample)/2
		z = block_size/2.-1.
		Fbin = float(Fsample)/block_size
		vh = ceil(Fbw_h/Fbin)
		vl = ceil(float(Signal_BW)/Fbin)
		vsig = ceil(float(Finput)/Fbin)
		s_width = 40000./block_num
		rsw = round(s_width/2.)
		Sig_pure = abs(self.Yp[(z+vsig-rsw):(z+vsig+rsw)])
		Sig_dB = 10*log10(sum(Sig_pure))
		T1 = abs(self.Yp[(z+vl):(z+vsig-rsw)])
		NT1=sum(T1)
		#****************** noise between signal & Fbw_h ******************
		NTother_abs=abs(self.Yp[z+vsig+rsw:z+vh])
		NT=NT1+sum(NTother_abs) 
		NT=10*log10(NT)    
		self.nt = Sig_dB

	def calc_sigdb(self):
		#****************** Post Process Data to get SNR ******************
		if len(self.data_file) == 1: # no data file
			return -(arange(-10., 10., 1.))**2		
		y = self.data_file		
		B = y-self.y_offset		
		Fbw_h=float(self.Fsample)/2
		#Ts=1/self.Fsample;
		z = self.block_size/2.-1.
		Fbin = float(self.Fsample)/self.block_size
		vh = ceil(Fbw_h/Fbin)
		vl = ceil(float(self.Signal_BW)/Fbin)
		vsig = ceil(float(self.Finput)/Fbin)
		s_width = 40000./self.AVG
		rsw = round(s_width/2.)
		Sig_pure = abs(self.Yp[(z+vsig-rsw):(z+vsig+rsw)])
		Sig_dB = 10*log10(sum(Sig_pure))
		T1 = abs(self.Yp[(z+vl):(z+vsig-rsw)])
		NT1=sum(T1)
		#****************** noise between signal & Fbw_h ******************
		NTother_abs=abs(self.Yp[z+vsig+rsw:z+vh])
		NT=NT1+sum(NTother_abs)  
		NT=10*log10(NT)    
		self.sigdb = (20*log10(max(B)/self.y_offset))	
        '''

	def calc_snrnew(self,data,Yf_linear,Finput,Signal_BW,f,Integ_low,Integ_high,full_scale,time_peakToPeak):
		
		if len(data) == 1: # no data file
			return -(arange(-10., 10., 1.))**2		
		f_signal_low = float(Finput) - float(Signal_BW)
		f_signal_high = float(Finput) + float(Signal_BW)
		index_signal_low = argmin(abs(f-f_signal_low))
		index_signal_high = argmin(abs(f-f_signal_high))
		self.f_signal_low = f_signal_low
		self.f_signal_high = f_signal_high
		sig_power = sum(abs(Yf_linear[index_signal_low:index_signal_high])**2)
                sig_power_db = 10*log10(sig_power)
                
		f_Integ_low = float(Integ_low)
		f_Integ_high = float(Integ_high)
		index_noise_low = argmin(abs(f-f_Integ_low))
		index_noise_high = argmin(abs(f-f_Integ_high))		
		#noise_power = sum(abs(Yf_linear[round(len(Yf_linear)/2.)+8:index_noise_low])**2) + sum(abs(Yf_linear[index_noise_high:])**2) # start from 1 - we don't want DC
                noise_power = sum(abs(Yf_linear[index_noise_low:index_signal_low])**2)+sum(abs(Yf_linear[index_signal_high:index_noise_high])**2)
                noise_power_db = 10*log10(noise_power)
                
                snr_float = 10.*log10(sig_power/noise_power)
                extra_snr = 20.*log10(full_scale/float(time_peakToPeak))
                snr = self.toEngFormat(snr_float)
                extra_snr = self.toEngFormat(extra_snr)                
				
		snrnew = str(snr)+'+'+str(extra_snr)
		
		return [snrnew,snr_float,sig_power_db,noise_power_db]
		
		'''
		#****************** Post Process Data to get SNR ******************
		if len(self.data_file) == 1: # no data file
			return -(arange(-10., 10., 1.))**2		
		Yf = self.Yf
		yf_linear = self._yf_linear[len(self._yf_linear)/2:]
		freq = self.f[len(self._yf_linear)/2:]		
		Fbw_h=float(self.Fsample)/2
		Ts=1/float(self.Fsample);
		half_fs = self.block_size/2.-1.
		Fbin = float(self.Fsample)/self.block_size
		vsig = ceil(float(self.Finput)/Fbin)
		vl = vsig - ceil(float(self.Signal_BW)/Fbin)
		vh = vsig + ceil(float(self.Signal_BW)/Fbin)
		sig_power = sum(abs(yf_linear[vl:vh])**2)
		print 'sig_power', sig_power, 'max_sig', max(yf_linear[vl:vh]),vl, vh
		noise_power = sum(abs(yf_linear[8:vl])**2) + sum(abs(yf_linear[vh:half_fs])**2) # start from 1 - we don't want DC
		print 'noise_power', noise_power, 'max_noise', max(max(abs(yf_linear[8:vl])**2), max(abs(yf_linear[vh:half_fs]**2)))		
		self.snrnew = 10.*log10(sig_power/noise_power)    
	        '''
	        
	###########################################################################
	#############################  update for the signal from user input
	def updateFromUser(self, name, new):
	    self.recalculate_user()   

	def active_user(self):
	    self.recalculate_user()
	    self.initiate_plot_user()
	    self.time_plot_change_xlow_user()
	    self.time_plot_change_xhigh_user()
	    self.time_plot_change_ylow_user()
	    self.time_plot_change_yhigh_user()
	    self.fft_plot_change_xlow_user()
	    self.fft_plot_change_xhigh_user()
	    self.fft_plot_change_ylow_user()
	    self.fft_plot_change_yhigh_user()	 
	    
	def initiate_plot_user(self):
	    y = self.signal_fft
	    x = self.frequency
	    self.fft_plot = Plot(self.plot_data)
	    self.renderer = self.fft_plot.plot(("x", "y"), type='line', color='magenta')	  
	    self.fft_plot.index_axis.title = 'frequency [Hz]'
	    self.fft_plot.value_axis.title = 'amplitude [dBc]'
	    self.fft_plot.title="FFT"
	    t = self.time_evaled
	    amp = self.signal_evaled_non_win
	    self.time_plot = Plot(self.plot_data)
	    self.renderer = self.time_plot.plot(("t","amp"),type='line',color='magenta')	
	    self.time_plot.index_axis.title = 'time [s]'
	    self.time_plot.value_axis.title = 'amplitude []'
	    self.time_plot.title = "Time"
	    print "This is _time_plot_default"  	          
	    		
        def time_plot_change_xlow_user(self):
	    self.time_plot.index_range.low = self.time_plot_xlow_user
	    self.time_plot.request_redraw()
	def time_plot_change_xhigh_user(self):
	    self.time_plot.index_range.high = self.time_plot_xhigh_user	    
	    self.time_plot.request_redraw()        
        def time_plot_change_ylow_user(self):
            self.time_plot.value_range.low = self.time_plot_ylow_user
	    self.time_plot.request_redraw()	
	def time_plot_change_yhigh_user(self):
	    self.time_plot.value_range.high = self.time_plot_yhigh_user	
	    self.time_plot.request_redraw()		
	def fft_plot_change_xlow_user(self):
	    self.fft_plot.index_range.low = self.fft_plot_xlow_user
	    self.fft_plot.request_redraw()
	def fft_plot_change_xhigh_user(self):
	    self.fft_plot.index_range.high = self.fft_plot_xhigh_user	    
	    self.fft_plot.request_redraw()        
        def fft_plot_change_ylow_user(self):
            self.fft_plot.value_range.low = self.fft_plot_ylow_user
	    self.fft_plot.request_redraw()	
	def fft_plot_change_yhigh_user(self):
	    self.fft_plot.value_range.high = self.fft_plot_yhigh_user	
	    self.fft_plot.request_redraw()			
	
	def recalculate_user(self):
		import numpy as np
		global normal, flicker, startstopstep
		# prepare dictionary
		eval_dict = np.__dict__.copy()
		eval_dict.update(self.__dict__)
		eval_dict['startstopstep'] = startstopstep 
		eval_dict['normal'] = normal
		eval_dict['white'] = normal
		eval_dict['pink'] = flicker
		eval_dict['flicker'] = flicker
		self.load_the_variables_into_dict(eval_dict)

		try: # calculate time
			time_evaled = eval(self.time_user, eval_dict)
		except Exception, inst:
			self.status = str(inst)
			return # don't update anything

		# add time to the dictionary
		eval_dict['time_user'] = time_evaled

		print 'f0',eval_dict['f0']
		print 'anoise',eval_dict['anoise']
		print 'fsample',eval_dict['fsample']
		print 'ncycles',eval_dict['ncycles']
	
		try: # calculate signal
			signal_evaled = eval(self.signal, eval_dict)
			self.signal_evaled_non_win = eval(self.signal, eval_dict)
			signal_evaled /= len(signal_evaled)
			#win = np.kaiser(len(signal_evaled),self.kaiser_beta)
			signal_prewindow = 1.0*signal_evaled
			win = blackmanharris(len(signal_evaled))
			signal_evaled *= win
		except Exception, inst:
			self.status = str(inst)
			return # don't update anything

		## CHECK, let's measure signal/noise in the time domain.
		if TIME_DOMAIN_CHECK:
			pure_signal = eval('sin(2*pi*f0*time_user)', eval_dict)
			pure_signal_power = sum( abs(pure_signal)**2)
			noise = (signal_prewindow*len(signal_prewindow))-pure_signal
			noise_power = sum( abs(noise)**2)
			print 'pure_signal', pure_signal_power
			print 'noise', noise_power
			print 'S/N in time domain', 10.0*log10(pure_signal_power/noise_power)

		try: # calculate signal_bw
			signal_bw = eval(self.signal_bw, eval_dict)
		except Exception, inst:
			self.status = str(inst)
			return # don't update anything

		try: # calculate signal_frequency
			signal_frequency = eval(self.signal_frequency, eval_dict)
		except Exception, inst:
			self.status = str(inst)
			return # don't update anything

		# Calculate frequencies
		frequency_unfiltered = fftfreq(len(time_evaled), d = time_evaled[1]-time_evaled[0])


		# get the indices of positive freq.  - we are assuming a real valued signal so we can throw away negative frequencies
		positive_frequency_indices = (frequency_unfiltered > 0).copy()

		frequency = frequency_unfiltered[positive_frequency_indices]

		signal_fft = fft(signal_evaled)
		signal_fft_linear = signal_fft = signal_fft[positive_frequency_indices] # get just positive frequencies


		signal_fft = 10*log10(abs(signal_fft)**2)

		self.signal_evaled = signal_evaled
		self.time_evaled = time_evaled
		self.frequency = frequency
		self.signal_fft = signal_fft

		# update snr
		fbin = frequency[1]-frequency[0]
		fcenter = floor(signal_frequency/fbin)
		f_low = max(0, ceil(fcenter - signal_bw/2/fbin) )
		f_high = min(len(frequency)-1, ceil(fcenter + signal_bw/2/fbin) )

		sig_power = sum(abs(signal_fft_linear[f_low:f_high])**2)
		#TODO: Make algorithmic or user input way to exclude DC instead of hardwiring 8 below
		noise_power = sum(abs(signal_fft_linear[100:f_low])**2) + sum(abs(signal_fft_linear[f_high:])**2)  
		self.snr = 10.*log10(sig_power/noise_power)

		# update the plot
		self.plot_data.set_data('x', frequency) # go ahead and set the plot
		self.plot_data.set_data('y', signal_fft)
		self.plot_data.set_data('t',self.time_evaled)
		self.plot_data.set_data('amp',self.signal_evaled_non_win)
		
		self.status = '' # situation normal
		
		
	##############################################################################
	############################ detect the UI variable changes
	def load_the_variables_into_dict(self, eval_dict):
		instance_traits = self.the_variables_ui._instance_traits()

		for key in instance_traits:
			eval_dict[key] = getattr(self.the_variables_ui, key)

	def _add_var_changed(self):
		uv = UserVar()
		result = uv.configure_traits(kind='modal')
		if result:
			self.the_variables.append(uv)
			uv.set_notifier(self.on_user_var_changed)

	def _the_variables_items_changed(self):
		self.load_the_variables_ui()

	def load_the_variables_ui(self):
		new_ui = self.the_variables_ui.clone()

		for the_variable in self.the_variables:
			name = the_variable.variable
			value = the_variable.value
			low = the_variable.low
			high = the_variable.high
			slider = the_variable.slider
			if hasattr(new_ui, name): # already exists
				setattr(new_ui, name, value)
				new_ui.traits()[name].low = low
				new_ui.traits()[name].high = high
				new_ui.traits()[name].slider = slider
			else:
				new_ui.add_trait(name, Float(value, mytype=True, low=low, high=high, slider=slider))

			# now add notifier to let us know when the value changes
			new_ui.on_trait_change(self.updateFromUser, name)

		self.the_variables_ui = new_ui # forces a refresh in the main ui
        
        def on_user_var_changed(self, *args, **kwargs):
		self.load_the_variables_ui()

       
        
        @staticmethod
	def sci_formatter(value):
		return '%.2E' %value	
		
	@staticmethod
	def find_peaks(data):
	    peaks_index = []
	    for i in range(1,len(data)-1):
	        if (data[i]>data[i-1]) and (data[i]>data[i+1]):
	            peaks_index.append(i)
	    return peaks_index
	    
	
	#upperUnit = ['K','M','G','T','P','E','Z','Y'];
	#lowerUnit = ['m','u','n','p','f',''
	
	#@staticmethod
	def toEngFormat(self,value):
	    decimal.getcontext().prec = self.precision
	    temp = decimal.Decimal(value)
	    value_eng = temp.normalize().to_eng_string()
	    return value_eng 

	def updateEngFormat(self):
	    self.Fsample = self.toEngFormat(self.Fsample)
	    self.Finput = self.toEngFormat(self.Finput)
	    self.Signal_BW = self.toEngFormat(self.Signal_BW)
	    self.Integ_low = self.toEngFormat(self.Integ_low)
	    self.Integ_high = self.toEngFormat(self.Integ_high)	    
	    self.fft_plot_xlow_file = self.toEngFormat(self.fft_plot_xlow_file)
	    self.fft_plot_xhigh_file = self.toEngFormat(self.fft_plot_xhigh_file)
	    self.fft_plot_ylow_file = self.toEngFormat(self.fft_plot_ylow_file)
	    self.fft_plot_yhigh_file = self.toEngFormat(self.fft_plot_yhigh_file)
	    self.time_plot_xlow_file = self.toEngFormat(self.time_plot_xlow_file)
	    self.time_plot_xhigh_file = self.toEngFormat(self.time_plot_xhigh_file)
	    self.time_plot_ylow_file = self.toEngFormat(self.time_plot_ylow_file)
	    self.time_plot_yhigh_file = self.toEngFormat(self.time_plot_yhigh_file)
	    self.time_yMax = self.toEngFormat(str(self.time_yMax))
	    self.time_yMin = self.toEngFormat(str(self.time_yMin))
	    self.time_yAverage = self.toEngFormat(str(self.time_yAverage))
	    self.time_peakToPeak = self.toEngFormat(str(self.time_peakToPeak))
	    self.f_min = self.toEngFormat(str(self.f_min))
	    self.f_max = self.toEngFormat(str(self.f_max))
	    self.c1_ADC = self.toEngFormat(self.c1_ADC)
	    self.c2_ADC = self.toEngFormat(self.c2_ADC)
	    self.c3_ADC = self.toEngFormat(self.c3_ADC)	  
	    self.Fs_ADC = self.toEngFormat(self.Fs_ADC)
	    self.Fin_ADC = self.toEngFormat(self.Fin_ADC)
	    self.Integ_low_ADC = self.toEngFormat(self.Integ_low_ADC)
	    self.Integ_high_ADC = self.toEngFormat(self.Integ_high_ADC)
	    self.fft_low_index_ADC = self.toEngFormat(self.fft_low_index_ADC)
	    self.fft_high_index_ADC = self.toEngFormat(self.fft_high_index_ADC)  
	    self.Signal_BW_ADC = self.toEngFormat(self.Signal_BW_ADC)	    
	    
	def go(self):
		self.configure_traits()

		
		
if __name__ == "__main__":
	main = FFTutor()
	main.go()
