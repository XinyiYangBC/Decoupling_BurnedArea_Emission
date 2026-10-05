
import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

startyear_use       = 2002;    # for calculating and plotting
endyear_use         = 2022;


# ============ MK trend test ===========
def calculate_trend_line(series):
    x=np.array(range(1,len(series)+1))
    y = series#.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    trend_line = slope * x + intercept
    slope_per = slope/np.mean(y)
    return trend_line, p_value

# =============== BA ===================
def calculate_Regional_Annual_Mean(var_name):
    dir_file = f"/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_BA_m2_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] /1E6/1E6*100 # m to km to Mha
    region_C = C.sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_sum

# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
BA_data_use = []
for i in range(len(Var_name)):
    current_var_name = Var_name[i]
    BA_data_use.append(calculate_Regional_Annual_Mean(current_var_name))


# =============== Emission ===================
def calculate_Regional_Annual_Mean(var_name):
    dir_file = f"/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_CO2_by_gC_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] /1E15 # g C to Pg C
    region_C = C.sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_sum

# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
Emission_data_use = []
for i in range(len(Var_name)):
    current_var_name = Var_name[i]
    Emission_data_use.append(calculate_Regional_Annual_Mean(current_var_name))


Emission_data_use = np.array(Emission_data_use)
BA_data_use = np.array(BA_data_use)

EI_data_use = Emission_data_use*1e3/BA_data_use  #Tg C/ Mha


# == Trend ==
data_use = BA_data_use
trendline_use = []
p_use = []
for i in range(0, 7):  # Loop through indices 1 to 5
    ratio = data_use[i]
    slp,p= calculate_trend_line(ratio)
    trendline_use.append(slp)
    p_use.append(p)

BA_trendline_data_use = np.array(trendline_use)

data_use = EI_data_use
trendline_use = []
p_use = []
for i in range(0, 7):  # Loop through indices 1 to 5
    ratio = data_use[i]
    slp,p= calculate_trend_line(ratio)
    trendline_use.append(slp)
    p_use.append(p)

EI_trendline_data_use = np.array(trendline_use)

# == Trend ==
data_use = Emission_data_use
trendline_use = []
p_use = []
for i in range(0, 7):  # Loop through indices 1 to 5
    ratio = data_use[i]
    slp,p= calculate_trend_line(ratio)
    trendline_use.append(slp)
    p_use.append(p)

Emission_trendline_data_use = np.array(trendline_use)


# ========================= Plotting =========================

biome_indices = [0, 1, 3, 5]  # Change this if you want other biomes
biome_names = ['Global', 'SAVA', 'BORF', 'TEMF']  # Corresponding labels

plot_time_period = pd.date_range(start=str(startyear_use), end=str(endyear_use), freq='YS')


Data_use_colors = ['black','forestgreen','orangered']
# Data_use_colors = ['firebrick','forestgreen','darkorange']
Data_use_colors = ['black','red','darkorange']  # current best colcor

Data_use_colors = ['black','red','orange']
Data_use_colors = ['black','red','darkorange']
Data_use_colors = ['black','#7363A6','#D67A13']
Data_use_colors = ['black',"#800080",'#EE9A22']
Data_use_colors = ['black',"#800080",'orange']

Data_use_colors2 = ['black','black',  'black']
Data_use_colors2 = Data_use_colors

Data_use_colors3 = ['black']    # x axis color
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig, axs = plt.subplots(2, 2, figsize=(22, 16))

plt.subplots_adjust(wspace=0.08)  # Decrease to bring columns closer together (default is ~0.2)
plt.subplots_adjust(hspace=0.25)  # vertical space between rows


axs = axs.flatten()

line_width =4
line_width2 =3
for idx, biome_index in enumerate(biome_indices):
    Emission_data_use_plot = Emission_data_use[biome_index]
    BA_data_use_plot = BA_data_use[biome_index]
    EI_data_use_plot = EI_data_use[biome_index]
    BA_trendline_data_use_plot = BA_trendline_data_use[biome_index]
    EI_trendline_data_use_plot = EI_trendline_data_use[biome_index]
    Emission_trendline_data_use_plot = Emission_trendline_data_use[biome_index]

    # Data_use_colors = [ "black", "red","darkorange"]
    ax1 = axs[idx]

    fontsize=28

    if idx == 0:
        # ====================== AX1 =====================
        ax1.spines['left'].set_position(('outward', 0))  # Offset the second y-axis
        ax1.plot(plot_time_period, Emission_data_use_plot, label="CO2 Emission", marker='o', markersize=10, markerfacecolor=Data_use_colors[0], markeredgewidth=2, linestyle='-', linewidth=line_width, color=Data_use_colors[0])
        ax1.plot(plot_time_period, Emission_trendline_data_use_plot, marker='', markersize=10, markerfacecolor='none', markeredgewidth=2, linestyle='--', linewidth=line_width2, color=Data_use_colors[0])

        ax1.set_ylim([0, 8.0])
        yticks = np.arange(0, 8.1, 2)  
        ax1.set_yticks(yticks)
        #custom_yticklabels = [f'{double(y)}' for y in yticks]
        custom_yticklabels = [f'{float(y):.1f}' for y in yticks]
        ax1.set_yticklabels(custom_yticklabels, fontsize=fontsize, color=Data_use_colors2[0])#, fontweight='bold')
        ax1.set_ylabel(r'CO2 Emission (Pg C yr$^{{-1}}$)', fontsize=fontsize, color=Data_use_colors2[0])# fontweight='bold')
        #ax1.yaxis.set_minor_locator(AutoMinorLocator(2))  # Adds a minor tick between each major x-tick
        #ax1.set_xlabel('Year', fontsize=14, color='black')

        jump_num = 5
        xticks = plot_time_period[::jump_num]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([d.year for d in xticks], rotation=0, fontsize=fontsize, color=Data_use_colors3[0], fontweight='bold')
        #ax1.xaxis.set_minor_locator(AutoMinorLocator(jump_num))  # Adds a minor tick between each major x-tick
        ax1.set_facecolor('white')

        ax1.spines['left'].set_color(Data_use_colors2[0])  # Makes the left spine blue
        ax1.spines['left'].set_linewidth(3)
        ax1.tick_params(axis='y', which='major', length=10, width=2, color =Data_use_colors2[0] )  # Increase size and thickness of major ticks


        ax1.spines['top'].set_visible(False)  # Hide the top spine
        ax1.spines['right'].set_visible(False)  
        #ax1.spines['left'].set_visible(False)  

        ax1.spines['bottom'].set_position(('outward', 10))  # Move the x-axis outward
        ax1.spines['bottom'].set_linewidth(2)  # Adjust the line width of the spine
        ax1.tick_params(axis='x', which='major', length=10, width=2, color=Data_use_colors3[0], labelsize=fontsize, labelcolor=Data_use_colors3[0])
        ax1.tick_params(axis='x', which='minor', length=5, width=1, color=Data_use_colors3[0])  # Optional minor ticks



        #ax1.set_title('Carbon', fontsize=16, fontweight='bold')
        #ax1.legend(loc='upper right', fontsize=18)

        # ax1.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.2)
        # ax1.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        # ax1.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
        #ax1.minorticks_on()

        # ====================== AX2 =====================
        ax2 = ax1.twinx()
        ax2.spines['right'].set_position(('outward', -5))  # Offset the second y-axis
        ax2.plot(plot_time_period, BA_data_use_plot/1e2, label="Burned Area", marker='o', markersize=10, markerfacecolor=Data_use_colors[1], markeredgewidth=2, linestyle='-', linewidth=line_width, color=Data_use_colors[1])
        ax2.plot(plot_time_period, BA_trendline_data_use_plot/1e2, marker='', markersize=10, markerfacecolor='none', markeredgewidth=2, linestyle='--', linewidth=line_width2, color=Data_use_colors[1])

        ax2.set_ylim([0, 15.00])
        yticks = np.arange(0, 15.1, 3.00)  
        ax2.set_yticks(yticks)
        #custom_yticklabels = [f'{double(y)}' for y in yticks]
        custom_yticklabels = [f'{float(y):.0f}' for y in yticks]
        ax2.set_yticklabels(custom_yticklabels, fontsize=fontsize, color=Data_use_colors2[1])# fontweight='bold')
        ax2.set_ylabel(r'Burned Area (×$10^2$ Mha yr$^{{-1}}$)', fontsize=fontsize, color=Data_use_colors2[1])# fontweight='bold')

        ax2.spines['top'].set_visible(False)  # Hide the top spine
        ax2.spines['left'].set_visible(False)  
        ax2.spines['bottom'].set_visible(False) 
        ax2.spines['right'].set_color(Data_use_colors2[1]) 

        ax2.spines['right'].set_linewidth(3)
        ax2.tick_params(axis='y', which='major', length=10, width=2, color =Data_use_colors2[1] )  # Increase size and thickness of major ticks


        #ax2.yaxis.set_minor_locator(AutoMinorLocator(2))  # Adds a minor tick between each major x-tick
        #ax2.set_xlabel('Year', fontsize=14, color='black')
        # xticks = plot_time_period[::4]
        # ax2.set_xticks(xticks)
        # ax2.set_xticklabels([d.year for d in xticks], rotation=0, fontsize=18, color='gray', fontweight='bold')
        # ax2.xaxis.set_minor_locator(AutoMinorLocator(2))  # Adds a minor tick between each major x-tick
        # ax2.set_facecolor('white')

        # ax2.legend(loc='upper right', fontsize=18)

        # ax2.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.2)
        # ax2.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        # ax2.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
        # ax2.minorticks_on()

        # ====================== AX3 =====================
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 82))  # Offset the second y-axis
        ax3.plot(plot_time_period, EI_data_use_plot, label="Emission Intensity", marker='o', markersize=10, markerfacecolor=Data_use_colors[2], markeredgewidth=2, linestyle='-', linewidth=line_width, color=Data_use_colors[2])
        ax3.plot(plot_time_period, EI_trendline_data_use_plot, marker='', markersize=10, markerfacecolor='none', markeredgewidth=2, linestyle='--', linewidth=line_width2, color=Data_use_colors[2])

        ax3.set_ylim([0.0, 5.0])
        yticks = np.arange(0.0, 5.1, 0.5*2)  
        ax3.set_yticks(yticks)
        #custom_yticklabels = [f'{double(y)}' for y in yticks]
        custom_yticklabels = [f'{float(y):.0f}' for y in yticks]
        ax3.set_yticklabels(custom_yticklabels, fontsize=fontsize, color=Data_use_colors2[2])# fontweight='bold')
        ax3.set_ylabel(r'EI (Tg C/Mha)', fontsize=fontsize, color=Data_use_colors2[2])# fontweight='bold')

        ax3.spines['top'].set_visible(False)  # Hide the top spine
        ax3.spines['left'].set_visible(False)  
        ax3.spines['bottom'].set_visible(False)  
        ax3.spines['right'].set_color(Data_use_colors2[2])  

        ax3.spines['right'].set_linewidth(3)
        ax3.tick_params(axis='y', which='major', length=10, width=2, color =Data_use_colors2[2] )  # Increase size and thickness of major ticks


        # adding texts 
        # ax1.text(plot_time_period[4]-pd.Timedelta(days=1100), 2.1, f'Emission: {-0.26:.2f}% yr$^{{-1}}$, p>0.1', color=Data_use_colors[0], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[4]-pd.Timedelta(days=600), 4.6, f'BA: {-1.21:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[1], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[4]-pd.Timedelta(days=600), 5.35, f'EI: +{0.95:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[2], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 1.2, f'Emission: -{0.26:.2f}% yr$^{{-1}}$, p>0.1', color=Data_use_colors[0], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 0.7, f'BA:           -{1.21:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[1], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 0.2, f'EI:            +{0.95:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[2], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        ax1.text(plot_time_period[8]-pd.Timedelta(days=00), 2.0, f'-{0.29:.2f}% yr$^{{-1}}$', color=Data_use_colors[0], fontsize=fontsize-2, verticalalignment='bottom')#, fontweight='bold')
        ax1.text(plot_time_period[8]-pd.Timedelta(days=00), 4.4, f'-{1.22:.2f}% yr$^{{-1}}$ **', color=Data_use_colors[1], fontsize=fontsize-2, verticalalignment='bottom')#, fontweight='bold')
        ax1.text(plot_time_period[8]-pd.Timedelta(days=00), 5.3, f'+{0.92:.2f}% yr$^{{-1}}$ **', color=Data_use_colors[2], fontsize=fontsize-2, verticalalignment='bottom')#, fontweight='bold')
                
        ax1.set_title(biome_names[idx], fontsize=27)#, fontweight='bold')
        #fig.text(0.1, 0.95, 'A', fontsize=40, color = 'Black', fontweight='bold', transform=fig.transFigure)


        # Special Setting
        # ax2.spines['right'].set_visible(False)  # hide right spine
        # ax2.tick_params(right=False, labelright=False)  # hide ticks and labels
        # ax2.set_ylabel("")  # remove y-axis label

        # ax3.spines['right'].set_visible(False)  # hide right spine
        # ax3.tick_params(right=False, labelright=False)  # hide ticks and labels
        # ax3.set_ylabel("")  # remove y-axis label

        # Combine legend handles and labels from all axes
        lines_labels1 = ax1.get_legend_handles_labels()
        lines_labels2 = ax2.get_legend_handles_labels()
        lines_labels3 = ax3.get_legend_handles_labels()

        # Concatenate the handles and labels
        all_handles = lines_labels1[0] + lines_labels2[0] + lines_labels3[0]
        all_labels = lines_labels1[1] + lines_labels2[1] + lines_labels3[1]

        # ====== Fine-tune Global subplot position to avoid tick collision ======
        pos = ax1.get_position()
        ax1.set_position([
        pos.x0- 0.015,     # shift a bit to the left
        pos.y0,
        pos.width *0.75,  # slightly squeeze width
        pos.height
        ])


    elif idx == 1:
        # ====================== AX1 =====================
        ax1.spines['left'].set_position(('outward', 0))  # Offset the second y-axis
        ax1.plot(plot_time_period, Emission_data_use_plot, label="CO2 Emission", marker='o', markersize=10, markerfacecolor=Data_use_colors[0], markeredgewidth=2, linestyle='-', linewidth=line_width, color=Data_use_colors[0])
        ax1.plot(plot_time_period, Emission_trendline_data_use_plot, marker='', markersize=10, markerfacecolor='none', markeredgewidth=2, linestyle='--', linewidth=line_width2, color=Data_use_colors[0])

        ax1.set_ylim([0, 8.0])
        yticks = np.arange(0, 8.1, 2)  
        ax1.set_yticks(yticks)
        #custom_yticklabels = [f'{double(y)}' for y in yticks]
        custom_yticklabels = [f'{float(y):.1f}' for y in yticks]
        ax1.set_yticklabels(custom_yticklabels, fontsize=fontsize, color=Data_use_colors2[0])# fontweight='bold')
        ax1.set_ylabel(r'CO2 Emission (Pg C yr$^{{-1}}$)', fontsize=fontsize, color=Data_use_colors2[0])# fontweight='bold')
        #ax1.yaxis.set_minor_locator(AutoMinorLocator(2))  # Adds a minor tick between each major x-tick
        #ax1.set_xlabel('Year', fontsize=14, color='black')

        # jump_num = 6
        xticks = plot_time_period[::jump_num]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([d.year for d in xticks], rotation=0, fontsize=fontsize, color='black', fontweight='bold')
        #ax1.xaxis.set_minor_locator(AutoMinorLocator(jump_num))  # Adds a minor tick between each major x-tick
        ax1.set_facecolor('white')

        ax1.spines['left'].set_color(Data_use_colors2[0])  # Makes the left spine blue
        ax1.spines['left'].set_linewidth(3)
        ax1.tick_params(axis='y', which='major', length=10, width=2, color =Data_use_colors2[0] )  # Increase size and thickness of major ticks


        ax1.spines['top'].set_visible(False)  # Hide the top spine
        ax1.spines['right'].set_visible(False)  
        # ax1.spines['left'].set_visible(False)
        #ax1.spines['left'].set_visible(False)  

        ax1.spines['bottom'].set_position(('outward', 10))  # Move the x-axis outward
        ax1.spines['bottom'].set_linewidth(2)  # Adjust the line width of the spine

        ax1.tick_params(axis='x', which='major', length=10, width=2, color=Data_use_colors3[0], labelsize=fontsize, labelcolor=Data_use_colors3[0])
        ax1.tick_params(axis='x', which='minor', length=5, width=1, color=Data_use_colors3[0])  # Optional minor ticks

        #ax1.set_title('Carbon', fontsize=16, fontweight='bold')
        #ax1.legend(loc='upper right', fontsize=18)

        # ax1.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.2)
        # ax1.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        # ax1.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
        #ax1.minorticks_on()

        # ====================== AX2 =====================
        ax2 = ax1.twinx()
        ax2.spines['right'].set_position(('outward', -5))  # Offset the second y-axis
        ax2.plot(plot_time_period, BA_data_use_plot/1e2, label="Burned Area", marker='o', markersize=10, markerfacecolor=Data_use_colors[1], markeredgewidth=2, linestyle='-', linewidth=line_width, color=Data_use_colors[1])
        ax2.plot(plot_time_period, BA_trendline_data_use_plot/1e2, marker='', markersize=10, markerfacecolor='none', markeredgewidth=2, linestyle='--', linewidth=line_width2, color=Data_use_colors[1])

        ax2.set_ylim([0, 15.00])
        yticks = np.arange(0, 15.1, 3.00)  
        ax2.set_yticks(yticks)
        #custom_yticklabels = [f'{double(y)}' for y in yticks]
        custom_yticklabels = [f'{float(y):.0f}' for y in yticks]
        ax2.set_yticklabels(custom_yticklabels, fontsize=fontsize, color=Data_use_colors2[1])# fontweight='bold')
        ax2.set_ylabel(r'Burned Area (×$10^2$ Mha yr$^{{-1}}$)', fontsize=fontsize, color=Data_use_colors2[1])# fontweight='bold')

        ax2.spines['top'].set_visible(False)  # Hide the top spine
        ax2.spines['left'].set_visible(False)  
        ax2.spines['bottom'].set_visible(False) 
        ax2.spines['right'].set_color(Data_use_colors2[1]) 

        ax2.spines['right'].set_linewidth(3)
        ax2.tick_params(axis='y', which='major', length=10, width=2, color =Data_use_colors2[1] )  # Increase size and thickness of major ticks


        #ax2.yaxis.set_minor_locator(AutoMinorLocator(2))  # Adds a minor tick between each major x-tick
        #ax2.set_xlabel('Year', fontsize=14, color='black')
        # xticks = plot_time_period[::4]
        # ax2.set_xticks(xticks)
        # ax2.set_xticklabels([d.year for d in xticks], rotation=0, fontsize=18, color='gray', fontweight='bold')
        # ax2.xaxis.set_minor_locator(AutoMinorLocator(2))  # Adds a minor tick between each major x-tick
        # ax2.set_facecolor('white')

        # ax2.legend(loc='upper right', fontsize=18)

        # ax2.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.2)
        # ax2.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        # ax2.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
        # ax2.minorticks_on()

        # ====================== AX3 =====================
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 82))  # Offset the second y-axis
        ax3.plot(plot_time_period, EI_data_use_plot, label="Emission Intensity", marker='o', markersize=10, markerfacecolor=Data_use_colors[2], markeredgewidth=2, linestyle='-', linewidth=line_width, color=Data_use_colors[2])
        ax3.plot(plot_time_period, EI_trendline_data_use_plot, marker='', markersize=10, markerfacecolor='none', markeredgewidth=2, linestyle='--', linewidth=line_width2, color=Data_use_colors[2])

        ax3.set_ylim([0, 4.])
        yticks = np.arange(0, 4.1, 1)  
        ax3.set_yticks(yticks)
        #custom_yticklabels = [f'{double(y)}' for y in yticks]
        custom_yticklabels = [f'{float(y):.1f}' for y in yticks]
        ax3.set_yticklabels(custom_yticklabels, fontsize=fontsize, color=Data_use_colors2[2])# fontweight='bold')
        ax3.set_ylabel(r'EI (Tg C/Mha)', fontsize=fontsize, color=Data_use_colors2[2])#fontweight='bold')

        ax3.spines['top'].set_visible(False)  # Hide the top spine
        ax3.spines['left'].set_visible(False)  
        ax3.spines['bottom'].set_visible(False)  
        ax3.spines['right'].set_color(Data_use_colors2[2])  

        ax3.spines['right'].set_linewidth(3)
        ax3.tick_params(axis='y', which='major', length=10, width=2, color =Data_use_colors2[2] )  # Increase size and thickness of major ticks


        # adding texts 
        # ax1.text(plot_time_period[4]-pd.Timedelta(days=1100), 1.5, f'Emission: {-0.35:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[0], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[4]-pd.Timedelta(days=600), 3.9, f'BA: {-1.07:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[1], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[4]-pd.Timedelta(days=600), 6.3, f'EI: +{0.72:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[2], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 1.2, f'Emission: -{0.35:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[0], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 0.7, f'BA:           -{1.07:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[1], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 0.2, f'EI:            +{0.72:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[2], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        ax1.text(plot_time_period[8]-pd.Timedelta(days=00), 1.5, f'-{0.38:.2f}% yr$^{{-1}}$ **', color=Data_use_colors[0], fontsize=fontsize-2, verticalalignment='bottom')#, fontweight='bold')
        ax1.text(plot_time_period[8]-pd.Timedelta(days=00), 3.9, f'-{1.07:.2f}% yr$^{{-1}}$ **', color=Data_use_colors[1], fontsize=fontsize-2, verticalalignment='bottom')#, fontweight='bold')
        ax1.text(plot_time_period[8]-pd.Timedelta(days=00), 6., f'+{0.68:.2f}% yr$^{{-1}}$ **', color=Data_use_colors[2], fontsize=fontsize-2, verticalalignment='bottom')#, fontweight='bold')
                
        ax1.set_title(biome_names[idx], fontsize=27)#, fontweight='bold')
        #fig.text(0.1, 0.95, 'A', fontsize=40, color = 'Black', fontweight='bold', transform=fig.transFigure)

        # Special Setting
        # ax1.spines['left'].set_visible(False)  # hide right spine
        # ax1.tick_params(left=False, labelleft=False)  # hide ticks and labels
        # ax1.set_ylabel("")  # remove y-axis label

        # Combine legend handles and labels from all axes
        lines_labels1 = ax1.get_legend_handles_labels()
        lines_labels2 = ax2.get_legend_handles_labels()
        lines_labels3 = ax3.get_legend_handles_labels()

        # Concatenate the handles and labels
        all_handles = lines_labels1[0] + lines_labels2[0] + lines_labels3[0]
        all_labels = lines_labels1[1] + lines_labels2[1] + lines_labels3[1]

        # ====== Fine-tune Global subplot position to avoid tick collision ======
        pos = ax1.get_position()
        ax1.set_position([
        pos.x0+ 0.028,     # shift a bit to the left
        pos.y0,
        pos.width *0.75,  # slightly squeeze width
        pos.height
        ])


    elif idx == 2:
        # ====================== AX1 =====================
        ax1.spines['left'].set_position(('outward', 0))  # Offset the second y-axis
        ax1.plot(plot_time_period, Emission_data_use_plot, label="CO2 Emission", marker='o', markersize=10, markerfacecolor=Data_use_colors[0], markeredgewidth=2, linestyle='-', linewidth=line_width, color=Data_use_colors[0])
        ax1.plot(plot_time_period, Emission_trendline_data_use_plot, marker='', markersize=10, markerfacecolor='none', markeredgewidth=2, linestyle='--', linewidth=line_width2, color=Data_use_colors[0])

        ax1.set_ylim([0, 1.4])
        yticks = np.arange(0, 1.41, 0.2)  
        ax1.set_yticks(yticks)
        #custom_yticklabels = [f'{double(y)}' for y in yticks]
        custom_yticklabels = [f'{float(y):.1f}' for y in yticks]
        ax1.set_yticklabels(custom_yticklabels, fontsize=fontsize, color=Data_use_colors2[0])# fontweight='bold')
        ax1.set_ylabel(r'CO2 Emission (Pg C yr$^{{-1}}$)', fontsize=fontsize, color=Data_use_colors2[0])# fontweight='bold')
        #ax1.yaxis.set_minor_locator(AutoMinorLocator(2))  # Adds a minor tick between each major x-tick
        #ax1.set_xlabel('Year', fontsize=14, color='black')

        # jump_num = 6
        xticks = plot_time_period[::jump_num]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([d.year for d in xticks], rotation=0, fontsize=fontsize, color='black', fontweight='bold')
        #ax1.xaxis.set_minor_locator(AutoMinorLocator(jump_num))  # Adds a minor tick between each major x-tick
        ax1.set_facecolor('white')

        ax1.spines['left'].set_color(Data_use_colors2[0])  # Makes the left spine blue
        ax1.spines['left'].set_linewidth(3)
        ax1.tick_params(axis='y', which='major', length=10, width=2, color =Data_use_colors2[0] )  # Increase size and thickness of major ticks


        ax1.spines['top'].set_visible(False)  # Hide the top spine
        ax1.spines['right'].set_visible(False)  
        #ax1.spines['left'].set_visible(False)  

        ax1.spines['bottom'].set_position(('outward', 10))  # Move the x-axis outward
        ax1.spines['bottom'].set_linewidth(2)  # Adjust the line width of the spine

        ax1.tick_params(axis='x', which='major', length=10, width=2, color=Data_use_colors3[0], labelsize=fontsize, labelcolor=Data_use_colors3[0])
        ax1.tick_params(axis='x', which='minor', length=5, width=1, color=Data_use_colors3[0])  # Optional minor ticks

        #ax1.set_title('Carbon', fontsize=16, fontweight='bold')
        #ax1.legend(loc='upper right', fontsize=18)

        # ax1.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.2)
        # ax1.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        # ax1.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
        #ax1.minorticks_on()

        # ====================== AX2 =====================
        ax2 = ax1.twinx()
        ax2.spines['right'].set_position(('outward', -5))  # Offset the second y-axis
        ax2.plot(plot_time_period, BA_data_use_plot/1e2, label="Burned Area", marker='o', markersize=10, markerfacecolor=Data_use_colors[1], markeredgewidth=2, linestyle='-', linewidth=line_width, color=Data_use_colors[1])
        ax2.plot(plot_time_period, BA_trendline_data_use_plot/1e2, marker='', markersize=10, markerfacecolor='none', markeredgewidth=2, linestyle='--', linewidth=line_width2, color=Data_use_colors[1])

        ax2.set_ylim([0, 1.20])
        yticks = np.arange(0, 1.21, 0.3)  
        ax2.set_yticks(yticks)
        #custom_yticklabels = [f'{double(y)}' for y in yticks]
        custom_yticklabels = [f'{float(y):.1f}' for y in yticks]
        ax2.set_yticklabels(custom_yticklabels, fontsize=fontsize, color=Data_use_colors2[1])# fontweight='bold')
        ax2.set_ylabel(r'Burned Area (×$10^2$ Mha yr$^{{-1}}$)', fontsize=fontsize, color=Data_use_colors2[1])# fontweight='bold')

        ax2.spines['top'].set_visible(False)  # Hide the top spine
        ax2.spines['left'].set_visible(False)  
        ax2.spines['bottom'].set_visible(False) 
        ax2.spines['right'].set_color(Data_use_colors2[1]) 

        ax2.spines['right'].set_linewidth(3)
        ax2.tick_params(axis='y', which='major', length=10, width=2, color =Data_use_colors2[1] )  # Increase size and thickness of major ticks


        #ax2.yaxis.set_minor_locator(AutoMinorLocator(2))  # Adds a minor tick between each major x-tick
        #ax2.set_xlabel('Year', fontsize=14, color='black')
        # xticks = plot_time_period[::4]
        # ax2.set_xticks(xticks)
        # ax2.set_xticklabels([d.year for d in xticks], rotation=0, fontsize=18, color='gray', fontweight='bold')
        # ax2.xaxis.set_minor_locator(AutoMinorLocator(2))  # Adds a minor tick between each major x-tick
        # ax2.set_facecolor('white')

        # ax2.legend(loc='upper right', fontsize=18)

        # ax2.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.2)
        # ax2.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        # ax2.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
        # ax2.minorticks_on()

        # ====================== AX3 =====================
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 82))  # Offset the second y-axis
        ax3.plot(plot_time_period, EI_data_use_plot, label="Emission Intensity", marker='o', markersize=10, markerfacecolor=Data_use_colors[2], markeredgewidth=2, linestyle='-', linewidth=line_width, color=Data_use_colors[2])
        ax3.plot(plot_time_period, EI_trendline_data_use_plot, marker='', markersize=10, markerfacecolor='none', markeredgewidth=2, linestyle='--', linewidth=line_width2, color=Data_use_colors[2])

        ax3.set_ylim([0.0, 35.0])
        yticks = np.arange(0.0, 35.1, 7)  
        ax3.set_yticks(yticks)
        #custom_yticklabels = [f'{double(y)}' for y in yticks]
        custom_yticklabels = [f'{float(y):.0f}' for y in yticks]
        ax3.set_yticklabels(custom_yticklabels, fontsize=fontsize, color=Data_use_colors2[2])# fontweight='bold')
        ax3.set_ylabel(r'EI (Tg C/Mha)', fontsize=fontsize, color=Data_use_colors2[2])# fontweight='bold')

        ax3.spines['top'].set_visible(False)  # Hide the top spine
        ax3.spines['left'].set_visible(False)  
        ax3.spines['bottom'].set_visible(False)  
        ax3.spines['right'].set_color(Data_use_colors2[2])  

        ax3.spines['right'].set_linewidth(3)
        ax3.tick_params(axis='y', which='major', length=10, width=2, color =Data_use_colors2[2] )  # Increase size and thickness of major ticks


        # adding texts 
        # ax1.text(plot_time_period[0]+pd.Timedelta(days=400), 0.52, f'Emission: +{1.06:.2f}% yr$^{{-1}}$, p>0.1', color=Data_use_colors[0], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[4]-pd.Timedelta(days=800), 0./1E2, f'BA: {-1.59:.2f}% yr$^{{-1}}$, p>0.1', color=Data_use_colors[1], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[4]-pd.Timedelta(days=800), 1.18, f'EI: +{2.31:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[2], fontsize=fontsize-2, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 1.25, f'Emission: +{1.06:.2f}% yr$^{{-1}}$, p>0.1', color=Data_use_colors[0], fontsize=fontsize-1, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 1.15, f'BA:           -{1.59:.2f}% yr$^{{-1}}$, p>0.1', color=Data_use_colors[1], fontsize=fontsize-1, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 1.05, f'EI:            +{2.31:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[2], fontsize=fontsize-1, verticalalignment='bottom', fontweight='bold')
        ax1.text(plot_time_period[12]-pd.Timedelta(days=00), 0.42, f'+{1.06:.2f}% yr$^{{-1}}$', color=Data_use_colors[0], fontsize=fontsize-2, verticalalignment='bottom')#, fontweight='bold')
        ax1.text(plot_time_period[12]-pd.Timedelta(days=00), 0.015, f'-{1.59:.2f}% yr$^{{-1}}$', color=Data_use_colors[1], fontsize=fontsize-2, verticalalignment='bottom')#, fontweight='bold')
        ax1.text(plot_time_period[12]-pd.Timedelta(days=00), 1.1, f'+{2.31:.2f}% yr$^{{-1}}$ **', color=Data_use_colors[2], fontsize=fontsize-2, verticalalignment='bottom')#, fontweight='bold')
              
        ax1.set_title(biome_names[idx], fontsize=27)#, fontweight='bold')
        #fig.text(0.1, 0.95, 'A', fontsize=40, color = 'Black', fontweight='bold', transform=fig.transFigure)


        # Special Setting
        # ax2.spines['right'].set_visible(False)  # hide right spine
        # ax2.tick_params(right=False, labelright=False)  # hide ticks and labels
        # ax2.set_ylabel("")  # remove y-axis label

        # ax3.spines['right'].set_visible(False)  # hide right spine
        # ax3.tick_params(right=False, labelright=False)  # hide ticks and labels
        # ax3.set_ylabel("")  # remove y-axis label

        # Combine legend handles and labels from all axes
        lines_labels1 = ax1.get_legend_handles_labels()
        lines_labels2 = ax2.get_legend_handles_labels()
        lines_labels3 = ax3.get_legend_handles_labels()

        # Concatenate the handles and labels
        all_handles = lines_labels1[0] + lines_labels2[0] + lines_labels3[0]
        all_labels = lines_labels1[1] + lines_labels2[1] + lines_labels3[1]

        # ====== Fine-tune Global subplot position to avoid tick collision ======
        pos = ax1.get_position()
        ax1.set_position([
        pos.x0- 0.015,     # shift a bit to the left
        pos.y0,
        pos.width *0.75,  # slightly squeeze width
        pos.height
        ])

    else:
        # ====================== AX1 =====================
        ax1.spines['left'].set_position(('outward', 0))  # Offset the second y-axis
        ax1.plot(plot_time_period, Emission_data_use_plot, label="CO2 Emission", marker='o', markersize=10, markerfacecolor=Data_use_colors[0], markeredgewidth=2, linestyle='-', linewidth=line_width, color=Data_use_colors[0])
        ax1.plot(plot_time_period, Emission_trendline_data_use_plot, marker='', markersize=10, markerfacecolor='none', markeredgewidth=2, linestyle='--', linewidth=line_width2, color=Data_use_colors[0])


        ax1.set_ylim([0, 0.4])
        yticks = np.arange(0, 0.41, 0.1)  
        ax1.set_yticks(yticks)
        #custom_yticklabels = [f'{double(y)}' for y in yticks]
        custom_yticklabels = [f'{float(y):.1f}' for y in yticks]
        ax1.set_yticklabels(custom_yticklabels, fontsize=fontsize, color=Data_use_colors2[0])# fontweight='bold')
        ax1.set_ylabel(r'CO2 Emission (Pg C yr$^{{-1}}$)', fontsize=fontsize, color=Data_use_colors2[0])# fontweight='bold')
        #ax1.yaxis.set_minor_locator(AutoMinorLocator(2))  # Adds a minor tick between each major x-tick
        #ax1.set_xlabel('Year', fontsize=14, color='black')


        # jump_num = 4
        xticks = plot_time_period[::jump_num]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([d.year for d in xticks], rotation=0, fontsize=fontsize, color='black', fontweight='bold')
        #ax1.xaxis.set_minor_locator(AutoMinorLocator(jump_num))  # Adds a minor tick between each major x-tick
        ax1.set_facecolor('white')

        ax1.spines['left'].set_color(Data_use_colors2[0])  # Makes the left spine blue
        ax1.spines['left'].set_linewidth(3)
        ax1.tick_params(axis='y', which='major', length=10, width=2, color =Data_use_colors2[0] )  # Increase size and thickness of major ticks


        ax1.spines['top'].set_visible(False)  # Hide the top spine
        ax1.spines['right'].set_visible(False)  
        # ax1.spines['left'].set_visible(False)
        #ax1.spines['left'].set_visible(False)  

        ax1.spines['bottom'].set_position(('outward', 10))  # Move the x-axis outward
        ax1.spines['bottom'].set_linewidth(2)  # Adjust the line width of the spine

        ax1.tick_params(axis='x', which='major', length=10, width=2, color=Data_use_colors3[0], labelsize=fontsize, labelcolor=Data_use_colors3[0])
        ax1.tick_params(axis='x', which='minor', length=5, width=1, color=Data_use_colors3[0])  # Optional minor ticks

        #ax1.set_title('Carbon', fontsize=16, fontweight='bold')
        #ax1.legend(loc='upper right', fontsize=18)

        # ax1.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.2)
        # ax1.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        # ax1.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
        #ax1.minorticks_on()

        # ====================== AX2 =====================
        ax2 = ax1.twinx()
        ax2.spines['right'].set_position(('outward', -5))  # Offset the second y-axis
        ax2.plot(plot_time_period, BA_data_use_plot/1e2, label="Burned Area", marker='o', markersize=10, markerfacecolor=Data_use_colors[1], markeredgewidth=2, linestyle='-', linewidth=line_width, color=Data_use_colors[1])
        ax2.plot(plot_time_period, BA_trendline_data_use_plot/1e2, marker='', markersize=10, markerfacecolor='none', markeredgewidth=2, linestyle='--', linewidth=line_width2, color=Data_use_colors[1])

        ax2.set_ylim([0, 0.60])
        yticks = np.arange(0, 0.61, 0.1)  
        ax2.set_yticks(yticks)
        #custom_yticklabels = [f'{double(y)}' for y in yticks]
        custom_yticklabels = [f'{float(y):.1f}' for y in yticks]
        ax2.set_yticklabels(custom_yticklabels, fontsize=fontsize, color=Data_use_colors2[1])# fontweight='bold')
        ax2.set_ylabel(r'Burned Area (×$10^2$ Mha yr$^{{-1}}$)', fontsize=fontsize, color=Data_use_colors2[1])# fontweight='bold')

        ax2.spines['top'].set_visible(False)  # Hide the top spine
        ax2.spines['left'].set_visible(False)  
        ax2.spines['bottom'].set_visible(False) 
        ax2.spines['right'].set_color(Data_use_colors2[1]) 

        ax2.spines['right'].set_linewidth(3)
        ax2.tick_params(axis='y', which='major', length=10, width=2, color =Data_use_colors2[1] )  # Increase size and thickness of major ticks


        #ax2.yaxis.set_minor_locator(AutoMinorLocator(2))  # Adds a minor tick between each major x-tick
        #ax2.set_xlabel('Year', fontsize=14, color='black')
        # xticks = plot_time_period[::4]
        # ax2.set_xticks(xticks)
        # ax2.set_xticklabels([d.year for d in xticks], rotation=0, fontsize=18, color='gray', fontweight='bold')
        # ax2.xaxis.set_minor_locator(AutoMinorLocator(2))  # Adds a minor tick between each major x-tick
        # ax2.set_facecolor('white')

        # ax2.legend(loc='upper right', fontsize=18)

        # ax2.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.2)
        # ax2.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        # ax2.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
        # ax2.minorticks_on()

        # ====================== AX3 =====================
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 82))  # Offset the second y-axis
        ax3.plot(plot_time_period, EI_data_use_plot, label="Emission Intensity", marker='o', markersize=10, markerfacecolor=Data_use_colors[2], markeredgewidth=2, linestyle='-', linewidth=line_width, color=Data_use_colors[2])
        ax3.plot(plot_time_period, EI_trendline_data_use_plot, marker='', markersize=10, markerfacecolor='none', markeredgewidth=2, linestyle='--', linewidth=line_width2, color=Data_use_colors[2])

        ax3.set_ylim([0.0, 35.0])
        yticks = np.arange(0.0, 35.1,7 )  
        ax3.set_yticks(yticks)
        #custom_yticklabels = [f'{double(y)}' for y in yticks]
        custom_yticklabels = [f'{float(y):.0f}' for y in yticks]
        ax3.set_yticklabels(custom_yticklabels, fontsize=fontsize, color=Data_use_colors2[2])# fontweight='bold')
        ax3.set_ylabel(r'EI (Tg C/Mha)', fontsize=fontsize, color=Data_use_colors2[2])# fontweight='bold')

        ax3.spines['top'].set_visible(False)  # Hide the top spine
        ax3.spines['left'].set_visible(False)  
        ax3.spines['bottom'].set_visible(False)  
        ax3.spines['right'].set_color(Data_use_colors2[2])  

        ax3.spines['right'].set_linewidth(3)
        ax3.tick_params(axis='y', which='major', length=10, width=2, color =Data_use_colors2[2] )  # Increase size and thickness of major ticks


        # adding texts 
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 0.085, f'Emission: +{3.79:.2f}% yr$^{{-1}}$, p<0.1', color=Data_use_colors[0], fontsize=fontsize, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 0.057, f'BA: {-1.19:.2f}% yr$^{{-1}}$, p>0.1', color=Data_use_colors[1], fontsize=fontsize, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[1]-pd.Timedelta(days=00), 0.26, f'EI: +{4.32:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[2], fontsize=fontsize, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 0.36, f'Emission: +{3.79:.2f}% yr$^{{-1}}$, p<0.1', color=Data_use_colors[0], fontsize=fontsize-1, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 0.33, f'BA:           -{1.19:.2f}% yr$^{{-1}}$, p>0.1', color=Data_use_colors[1], fontsize=fontsize-1, verticalalignment='bottom', fontweight='bold')
        # ax1.text(plot_time_period[0]-pd.Timedelta(days=00), 0.30, f'EI:            +{4.32:.2f}% yr$^{{-1}}$, p<0.05', color=Data_use_colors[2], fontsize=fontsize-1, verticalalignment='bottom', fontweight='bold')
        ax1.text(plot_time_period[2]-pd.Timedelta(days=00), 0.06, f'+{3.86:.2f}% yr$^{{-1}}$ *', color=Data_use_colors[0], fontsize=fontsize-2, verticalalignment='bottom')#, fontweight='bold')
        ax1.text(plot_time_period[2]-pd.Timedelta(days=00), -0.012, f'-{1.19:.2f}% yr$^{{-1}}$', color=Data_use_colors[1], fontsize=fontsize-2, verticalalignment='bottom')#, fontweight='bold')
        ax1.text(plot_time_period[2]-pd.Timedelta(days=00), 0.14, f'+{4.39:.2f}% yr$^{{-1}}$ **', color=Data_use_colors[2], fontsize=fontsize-2, verticalalignment='bottom')#, fontweight='bold')
        
        ax1.set_title(biome_names[idx], fontsize=27)#, fontweight='bold')
        #fig.text(0.1, 0.95, 'A', fontsize=40, color = 'Black', fontweight='bold', transform=fig.transFigure)

        # # Special Setting
        # ax1.spines['left'].set_visible(False)  # hide right spine
        # ax1.tick_params(left=False, labelleft=False)  # hide ticks and labels
        # ax1.set_ylabel("")  # remove y-axis label

        # Combine legend handles and labels from all axes
        lines_labels1 = ax1.get_legend_handles_labels()
        lines_labels2 = ax2.get_legend_handles_labels()
        lines_labels3 = ax3.get_legend_handles_labels()

        # Concatenate the handles and labels
        all_handles = lines_labels1[0] + lines_labels2[0] + lines_labels3[0]
        all_labels = lines_labels1[1] + lines_labels2[1] + lines_labels3[1]

        # ====== Fine-tune Global subplot position to avoid tick collision ======
        pos = ax1.get_position()
        ax1.set_position([
        pos.x0+ 0.028,     # shift a bit to the left
        pos.y0,
        pos.width *0.75,  # slightly squeeze width
        pos.height
        ])

# Label subplots A, B, C, D
letters = ['a', 'b', 'c', 'd']
for i, ax in enumerate(axs):
    ax.text(-0.21, 1.1, letters[i], transform=ax.transAxes, fontsize=37, fontweight='bold', va='top')

# plt.tight_layout()
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/Figure_1_4regions_timeseries.tif', dpi=300, bbox_inches='tight')
plt.show()

