# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:18:56 2021

@author: asligar
"""

from read_frtm import read_frtm, get_results_files
import os
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
plt.close('all')

def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))


path = './example_frtm/'
#path = './example_frtm/Intersection.aedtexport/design1/Setup1_ts_8_18_2022_8_51_42_AM/'

show_rd = True
show_ra =True
show_peaks = True
ant_space = 0.73

results_dict = get_results_files(path)
num_results = len(results_dict)
t_start = 0
t_stop = 13
t_sweep= list(results_dict.keys())
fps = 1/(t_sweep[1]-t_sweep[0])
rPixels =256
dPixels =256
azPixels = 256

#read just the header of one file to get simulation parameters like num frequency points
#assumes data is the same for all files in the directory
all_results = []
all_results_ra = []
for t_val in results_dict.keys():
    data= read_frtm(results_dict[t_val])
    
    nfreq = data.nfreq
    fc = data.freq_center
    ntime = data.ntime
    
    
    r_period = data.range_period()
    r_resolution = data.range_resolution()
    range_vals = np.linspace(0,r_period,num=rPixels)
    v_period = data.velocity_period()
    v_resolution = data.velocity_resolution()
    vel_vals = np.linspace(-v_period/2,v_period/2,num=dPixels)
    cpi_time = data.time_duration
    prf = 1/data.time_delta
    vmin = -v_period/2
    vmax = v_period/2
    rmax = r_period
    fov = [-90,90]
    
    channel_names = data.channel_names
    num_channels = data.num_channels
    
    
    #order is if we want first index to be freq or pulse, post processing here assumes [freq][pulse] order
    single_frame = data.load_data(order='FreqPulse') 
    
    
    #################
    # Create range doppler plot for the first channel
    
    data_all_channels_rd = np.zeros((num_channels,rPixels,dPixels),dtype=complex)
    for n, ch in enumerate(channel_names):
        data_all_channels_rd[n], range_profile, processing_fps =utils.range_doppler_map(single_frame[ch], window=False,size=(rPixels,dPixels))
    
    
    #################
    # Create AoA plot for all range bins
    # this assumes the channels are correctly ordered, if not results will be wrong
    # re-order channels if needed

    # here I am just loading the results, which were in teh format [dict_key_channel_name][freq][pulse]
    # into a 3D array with format [channel_number][freq][pulse]. This just loads them in in the order
    # the keys appears. for MIMO or port names that are not sorted alphabetically you could modify this 
    data_all_channels_fp =np.zeros((num_channels,nfreq,ntime),dtype=complex)
    for n, ch in enumerate(channel_names):
        data_all_channels_fp[n] = single_frame[ch]

    #calculate range vs angle map across all range bins


    all_r_angle_data,aoa_processing_fps = utils.range_angle_map(data_all_channels_fp,
                                                                antenna_spacing_wl = ant_space,
                                                                source_data = 'FreqPulse',
                                                                DoA_method='bartlett',
                                                                fov = fov, 
                                                                out_size=(rPixels,azPixels))
    
    all_results_ra.append(all_r_angle_data) 
    #range doppler for a single channel
    #range_doppler = data_all_channels_rd[0][0:int(rPixels/2)]
    all_results.append(data_all_channels_rd)
  
#all_results = 20*np.log10(np.abs(np.array(all_results)))
all_results= np.array(all_results)
all_results_abs = np.abs(np.array(all_results))

all_results_ra = np.abs(np.array(all_results_ra))


if show_rd:
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(8,8) )
    
    a = 20*np.log10(np.abs(all_results[0][0]))

    im = plt.imshow(a, interpolation='hamming', aspect='auto',vmin=-150,vmax=-50)
    plt.ylabel('Range')
    plt.xlabel('Doppler')
    plt.title('Doppler Range')
    def update_plot_rd(i):
        print(t_sweep[i])
        val = 20*np.log10(np.abs(all_results[i][0]))
        print(np.max(val))
        im.set_array(val)
        return [im]
    
    anim = animation.FuncAnimation(
                                    fig, 
                                    update_plot_rd, 
                                    frames = num_results,
                                    interval = 1000 / fps, # in ms
                                    )







if show_ra:
    max_ra = np.max(20*np.log10(np.abs(all_results_ra)))
    dynamic_range = 140
    
    ra_to_plot1 = np.swapaxes(20*np.log10(np.abs(all_results_ra[0])),0,1)
    
    #normalize plot to maximum value so values stay constant for animation
    max_of_plot1 = max_ra
    min_of_plot1 = max_ra-dynamic_range
    levels = np.linspace(min_of_plot1,max_of_plot1,64)
    
    normi = colors.Normalize(vmin=min_of_plot1, vmax=max_of_plot1)
    
    azimuth_vals = np.linspace(fov[0],fov[1],num=azPixels)
    r, theta = np.meshgrid(range_vals,np.radians(azimuth_vals))
    
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    
    
    ax.set_thetamin((fov[0]))
    ax.set_thetamax((fov[1]))
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.set_ylim(0,np.max(range_vals))
    ax.set_yticks(np.arange(0,np.max(range_vals),10))
    ax.set_title('Angle vs Range (Azimuth)')
    #cbar = fig.colorbar(pp)
    #plt.show()
    
    
    
    
    def update_plot_ra(i):
        ra_to_plot1 = np.swapaxes(20*np.log10(np.abs(all_results_ra[i])),0,1)
        print(t_sweep[i])
        pp = plt.contourf(theta,r,ra_to_plot1,64,cmap='rainbow',
                levels=levels, norm=normi, extend='both')


    anim = animation.FuncAnimation(
                                    fig, 
                                    update_plot_ra, 
                                    frames = num_results,
                                    interval = 1000 / fps, # in ms
                                    )


# #calculate target list
#this can be used to plot any quanitity of detected peaks

all_targets = []
all_peaks = []
cumulative_peaks = None
all_xy = []
all_range_xrange = []

for idx in range(num_results):

    target_list, fps_target_list, peaks = utils.create_target_list(rd_all_channels_az =all_results[idx],
                                                        rd_all_channels_el = None,
                                                        rngDomain = range_vals,
                                                        velDomain = vel_vals,
                                                        azPixels=azPixels,
                                                        elPixels=0,
                                                        antenna_spacing_wl=ant_space,
                                                        radar_fov=fov,
                                                        centerFreq=fc,
                                                        rcs_min_detect=-10,
                                                        min_detect_range=1,
                                                        rel_peak_threshold = 1e-3,
                                                        max_detections=3,
                                                        return_cfar=True)
    all_targets.append(target_list)
    all_peaks.append(peaks)
    if len(target_list)>0:
        for tar in target_list:
            all_xy.append([target_list[tar]['xpos'],target_list[tar]['ypos']])
            all_range_xrange.append([target_list[tar]['range'],target_list[tar]['azimuth']])
    if cumulative_peaks is None:
        cumulative_peaks =peaks
    else:
        cumulative_peaks+=peaks



if show_peaks:
    
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(8,8) )
    
    a = all_peaks[0]
    im = plt.imshow(a, interpolation='hamming', aspect='auto')
    plt.ylabel('Range')
    plt.xlabel('Cross Range')
    plt.title('Doppler Range')
    def update_plot_peak(i):
        print(t_sweep[i])
        im.set_array(all_peaks[i])
        return [im]
    
    anim = animation.FuncAnimation(
                                    fig, 
                                    update_plot_peak, 
                                    frames = num_results,
                                    interval = 1000 / fps, # in ms
                                    )
    
    


    


# all_range_xrange = np.array(all_range_xrange)
# all_r = all_range_xrange[:,0]
# all_theta = all_range_xrange[:,1]*np.pi/180
# colors = np.zeros(len(all_theta))
# fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))


# ax.set_thetamin((fov[0]))
# ax.set_thetamax((fov[1]))
# ax.yaxis.grid(False)
# ax.xaxis.grid(False)
# ax.set_ylim(0,np.max(range_vals))
# ax.set_yticks(np.arange(0,np.max(range_vals),10))
# ax.set_title('Angle vs Range (Azimuth)')

# all_r_ideal = all_range_xrange_ideal[:,0]
# all_theta_ideal = all_range_xrange_ideal[:,1]*np.pi/180
# colors_ideal= np.ones(len(all_theta_ideal))
# colors2 = np.hstack((colors_ideal,colors))
# all_theta2 = np.hstack((all_theta_ideal,all_theta))
# all_r2 = np.hstack((all_r_ideal,all_r))
# c = ax.scatter(all_theta2, all_r2,c=colors2)

