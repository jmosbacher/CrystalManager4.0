import os
from traits.api import *
from traitsui.api import *
from traitsui.extras.checkbox_column import CheckboxColumn
from traitsui.ui_editors.data_frame_editor import DataFrameEditor
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from auxilary_functions import wl_to_rgb, bin_data_array, integrate_gaussian, gauss
import numpy as np
import random
import pandas as pd
from pandas.tools.plotting import lag_plot, autocorrelation_plot
from data_plot_viewers import SingleDataPlot
from fitting_tools import FittingTool1D
from plotting_tools import MeasurementPlottingTool
from analysis_tools import MeasurementAnalysisTool
import scipy.stats as stats
from pandas.tools.plotting import scatter_matrix
from datetime import datetime
try:
    import cPickle as pickle
except:
    import pickle

class BaseMeasurement(HasTraits):
    __kind__ = 'Base'
    main = Any()
    name = Str('Name')
    date = Date()
    time = Time()
    summary = Property(Str)

    notes = Str('')
    notebook = Int(1)
    page = Int()

    is_selected = Bool(False)

    def __init__(self, **kargs):
        HasTraits.__init__(self)
        self.main = kargs.get('main', None)

    def _anytrait_changed(self,name):
        if self.main is None:
            return
        if name in ['date', 'name', 'time', 'summary',
                        'notes', 'notebook', 'page',
                    'duration','ex_pol','em_pol','ex_wl',
                    'em_wl','exposure','frames','e_per_count',
                    'signal','bg','ref','file_data']:

            self.main.dirty = True

    def _get_summary(self):
        raise NotImplemented


class SpectrumMeasurement(BaseMeasurement):
    __kind__ = 'Spectrum'
    date = Property(Date)
    time = Property(Time)

    #####       User Input      #####
    duration = Property(Float) #Float(0)
    ex_pol = Int()  # Excitation Polarization
    em_pol = Int()  # Emission Polarization

    ex_wl = Float()  # Excitation Wavelength
    em_wl = Tuple((0.0, 0.0), labels=['Min', 'Max'])  # Emission Wavelength

    exposure = Property(Float)#Float(1)
    frames = Property(Int)#Int(1)
    e_per_count = Property(Int)#Int(1)  # electrons per ADC count

    #####       Extracted Data      #####
    data = Instance(pd.DataFrame)
    metadata = Dict()


    #####       Flags      #####
    has_signal = Property(Bool) #Bool(False)
    has_bgd = Property(Bool) #Bool(False)
    has_bgd_corrected = Property(Bool)
    has_ref = Property(Bool) #Bool(False)
    has_fits = Property(Bool)
    color = Property() #Tuple(0.0, 0.0, 0.0)  # Enum(['r', 'g', 'b', 'y', 'g', 'k','m','c','k'])
    #data_on = Bool(False)
    #####       Calculated Data      #####
    fits = List([])
    fit_data = Property(Array)
    resolution = Property()
    all_data_array = Array(transient=True)
    #show_hide_data = Button('Show/Hide Data')

    #####       UI      #####


    #####       GUI layout      #####
    plotting_tool = Instance(MeasurementPlottingTool,transient=True)
    analysis_tool = Instance(MeasurementAnalysisTool,transient=True)
    fitting_tool = Instance(FittingTool1D,transient=True)

    view = View(
        Tabbed(
        HGroup(
            VGroup(

            VGroup(
                #Item(name='ex_pol', label='Excitation POL'),
                Item(name='ex_wl', label='Excitation WL'),

                Item(name='frames', label='Frames'),
                Item(name='exposure', label='Exposure'),
                Item(name='color', label='Plot Color'),

                show_border=True, label='Excitation'),
            VGroup(
                #Item(name='em_pol', label='Emission POL'),
                Item(name='em_wl', label='Emission WL'),
                Item(name='e_per_count', label='e/count'),


                show_border=True, label='Emission'),
            #HGroup(Item(name='show_data', show_label=False, springy=True)),
            VGroup(
                    Item(name='metadata', editor=ValueEditor(), show_label=False),
                    label='Metadata'),
            springy=False),


                Item(name='data',show_label=False,
                     editor=DataFrameEditor(show_titles=True,
                                            #format='%g',
                                            show_index=False),

                #scrollable=True
                ),
        label='Data'),


            VGroup(
                Item(name='plotting_tool', show_label=False, style='custom', springy=False),

            label='Visualization'),

            VGroup(
                Item(name='fitting_tool', show_label=False, style='custom', springy=False),
                show_border=True, label='Fitting'),

            VGroup(
                Item(name='analysis_tool', show_label=False, style='custom', springy=False),
            label='Statistical Analysis'),

        ),
    )

    #####       Initialzation Methods      #####
    def _plotting_tool_default(self):
        return MeasurementPlottingTool(measurement=self)

    def _analysis_tool_default(self):
        return MeasurementAnalysisTool(measurement=self)

    def _fitting_tool_default(self):
        return FittingTool1D(measurements=[self],
                             name=self.name)
    def _data_default(self):
        return pd.DataFrame()

    #####       getters      #####
    def _get_summary(self):
        report = 'Excitation: %d nm'%self.ex_wl + ' | Emission Range: %d:%d nm'%self.em_wl
        return report

    @property_depends_on('metadata', settable=True)
    def _get_exposure(self):
        return float(self.metadata.get('sig',{}).get('Exposure Time (secs)',1))

    @property_depends_on('metadata', settable=True)
    def _get_frames(self):
        return int(self.metadata.get('sig',{}).get('Number of Accumulations',1))

    @property_depends_on('metadata', settable=True)
    def _get_date(self):
        try:
            str = self.metadata.get('sig', {}).get('Date and Time', ' ')
            date_time = datetime.strptime(str, '%a %b %d %X %Y')
            return date_time.date()
        except:
            return None

    @property_depends_on('metadata', settable=True)
    def _get_time(self):
        try:
            str = self.metadata.get('sig', {}).get('Date and Time', ' ')
            date_time = datetime.strptime(str, '%a %b %d %X %Y')
            return date_time.time()
        except:
            return None

    @property_depends_on('metadata', settable=True)
    def _get_e_per_count(self):
        preamp_gain = int(self.metadata.get('sig', {}).get('Pre-Amplifier Gain', '4'))
        amp_setting = self.metadata.get('sig', {}).get('Output Amplifier', 'High Sensitivity')
        camera_model = self.metadata.get('sig', {}).get('Model', 'DU940P')
        amp_gain = {'High Sensitivity':4, 'High Capacity':1}[amp_setting]

        return 16/(amp_gain*preamp_gain)


    @property_depends_on('ex_wl', settable=True)
    def _get_color(self):
        return wl_to_rgb(self.ex_wl)


    def _get_resolution(self):
        if self.has_signal:
            return self.data['em_wl'].diff().abs().mode()[0]
        else:
            return 0.0075

    def _get_has_signal(self):
        if 'sig' in self.data.columns.values():
            return True
        else:
            return False

    def _get_has_bgd(self):
        if 'bgd' in self.data.columns.values():
            return True
        else:
            return False

    def _get_has_ref(self):
        if 'ref' in self.data.columns.values():
            return True
        else:
            return False

    def _get_has_fits(self):
        if len(self.fits):
            return True
        else:
            return False

    def _data_changed(self):
        if 'em_wl' in self.data.columns.values:
            self.em_wl = self.data['em_wl'].min(),self.data['em_wl'].max()

    #####       Public Methods      #####
    def rescale(self, scale):
        cols = [col for col in self.data.columns.values if col != 'em_wl']
        self.data[cols] = self.data[cols]*scale



    def make_db_dataframe(self):
        final = self.data.copy()
        return final.set_index('em_wl')

    def normalized(self):
        cols = [col for col in self.data.columns.values if col != 'em_wl']
        normed = self.data.copy()
        normed[cols] = normed[cols]/(self.exposure*self.frames)
        return normed


    def bin_data(self,**kwargs):
        """
        :return:
        """

        normalize = kwargs.get('normalize',True)
        binsize = kwargs.get('binsize',0)

        if normalize:
            normed = self.normalized()
        else:
            normed = self.data

        if binsize:
            fact = binsize
        else:
            fact=int(1/self.resolution)

        binned = normed.groupby(normed//fact).sum()
        binned['em_wl'] = self.data['em_wl'].groupby(self.data.index//fact).mean()
        return binned


    def plot_data(self,**kwargs ):
        ax = kwargs.get('ax',None)
        legend = kwargs.get('legend',True)
        data_name = kwargs.get('data_name','bgd')
        title = kwargs.get('title',None)

        if self.has_signal:
            ser = pd.Series(data=self.data[data_name],index=self.data['em_wl'])
            axs = ser.plot(color=self.color, legend=legend, ax=ax)
            if ax is not None:
                if title is None:
                    ax.set_title(data_name,fontsize=12)
                ax.set_xlabel('Emission Wavelength')
                ax.set_ylabel('Counts')
                #plt.show()
            else:
                plt.show()
            return axs

    def plot_by_name(self,plot_name ='hist',title=None,data_name='bg_corrected', **kwargs):
        ax = kwargs.get('ax', None)
        #legend = kwargs.get('legend', True)
        #data = kwargs.get('data', 'bg_corrected')
        #title = kwargs.get('title', None)
        #alpha = kwargs.get('alpha', 1.0)
        if self.has_signal:
            ser = pd.Series(data=self.data[data_name],index=self.data['em_wl'])
            axs = getattr( ser.plot,plot_name)(color=self.color, **kwargs)
            if ax is not None:
                if title is None:
                    ax.set_title(data_name+' '+plot_name, fontsize=12)

            else:
                plt.show()
            return axs

    def plot_special(self,plot_name='autocorrelation',title=None,data_name ='bg_corrected', **kwargs):
        if not self.has_signal:
            return
        fig = kwargs.get('figure', plt.figure())
        ax = kwargs.get('ax', fig.add_subplot(111))
        nbins = kwargs.get('nbins',150)

        ser = pd.Series(data=self.data[data_name],index=self.data['em_wl'])
        #data_name = kwargs.get('data_name', 'BG corrected')
        axs = {
            'lag':lag_plot,
            'autocorrelation':autocorrelation_plot,
            }[plot_name](ser,**kwargs)
        if title is None:
            axs.set_title(' '.join([data_name,plot_name]) , fontsize=12)
        if ax is None:
            plt.show()

    def plot_scatter_matrix(self,**kwargs):
        diag = kwargs.get('diag','kde')
        if not self.has_signal:
            return
        fig = kwargs.get('figure', plt.figure())
        ax = kwargs.get('ax', fig.add_subplot(111))
        df = self.data
        scatter_matrix(df,ax=ax, diagonal=diag)

    def calc_statistics(self,statistic='hist',data_name='bg_corrected',**kwargs):

        ser = pd.Series(data=self.data[data_name],index=self.data['em_wl'])
        if statistic == 'hist':
            bins = kwargs.get('bins', 150)
            cnts, divs = np.histogram(ser,bins=bins)
            return pd.Series(data=cnts, index=divs[:-1]+np.diff(divs)/2, name=self.ex_wl)

        elif statistic == 'kde':
            nsample = kwargs.get('nsample', 200)
            arr = ser.as_matrix()
            kern = stats.gaussian_kde(arr)
            rng = np.linspace(arr.min(),arr.max(),nsample)
            return pd.Series(data=kern(rng), index=rng, name=self.ex_wl)


class AnealingMeasurement(BaseMeasurement):
    __kind__ = 'Anealing'
    temperature = Int(0)
    heating_time = Int(0)
    view = View(
        VGroup(

            HGroup(
                Item(name='temperature', label='Temperature'),
                Item(name='heating_time', label='Heating time'),
                show_border=True, label='Anealing Details'),

        ),

)


class SpectrumMeasurementSimulation(SpectrumMeasurement):
    simulation_data = Dict()

class MeasurementTableEditor(TableEditor):

    columns = [
               CheckboxColumn(name='is_selected', label='', width=0.08, horizontal_alignment='center',editable=True ),
               ObjectColumn(name='name', label='Name', horizontal_alignment='left', width=0.3,editable=True ),
               ObjectColumn(name='summary', label='Details', width=0.55, horizontal_alignment='center',editable=False  ),
               ObjectColumn(name='date', label='Date', horizontal_alignment='left', width=0.18,editable=False),
               ObjectColumn(name='__kind__', label='Type', width=0.18, horizontal_alignment='center',editable=False),


               ]

    auto_size = False
    sortable = False
    editable = True
    #scrollable=False
