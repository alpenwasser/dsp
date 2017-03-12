#!/usr/bin/env python3

from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

# Performance
# Desktop: 2 x Intel Xeon X5680 (3.33 GHz, 2 x 6PC/12LC == 12PC/24LC):
# 1e6 analog points: 5349.97s user 23.66s system 2093% cpu 4:16.65  total
# 1e5 analog points:  548.00s user  2.25s system 2132% cpu   25.797 total
# 1e4 analog points:   54.54s user  0.23s system 1765% cpu    3.102 total
# 1e6 analog points, 21 freq factor, trapezoid wave:
#                   34559.47s user 53.52s system 2250% cpu 25:38.09 total
#                   memory usage: about 14 GB

# Macbook Pro 2014: Intel Core I7 4770HQ (3.2 GHz turbo, 4PC/8LC):
# 1e6 analog points: 3426.85s user 14.19s system 769% cpu 7:26.98  total
# 1e5 analog points:  295.04s user  0.85s system 767% cpu   38.578 total
# 1e4 analog points:   27.02s use r 0.16s system 691% cpu    3.931 total


# ########################################################################## #
# GLOBAL CONSTANTS                                                           #
# ########################################################################## #
# -------------------------------------------------------- #
# For  good results,  ensure that  the evaluated  range is #
# a  good  bit  longer  than  the range  which  is  to  be #
# plotted. Default is a factor or 10.                      #
# -------------------------------------------------------- #
PLOT_START  =  9 * np.pi
PLOT_STOP   = 11 * np.pi
T_START     =  0
T_STOP      = PLOT_STOP + (PLOT_START - T_START)
# -------------------------------------------------------- #
# NOTE: If the number of analog points is too low, some of #
# the  dirac pulses  will  coincide. This  means that  the #
# pulse sequence at those points will have a value of 2 or #
# higher, instead of 1.                                    #
# Therefore,  ensure that  NA  is  high enough. Roughly  a #
# thousand  samples  per  2*pi  of  evaluated  range  (the #
# difference between T_START and T_STOP) are a pretty good #
# value in my experience.                                  #
# A higher value will result in a smaller error signal.    #
# -------------------------------------------------------- #
NA          = 1e6
# -------------------------------------------------------- #
# The  number of  points to  be plotted  shouldn't be  too #
# high. NA and PLOTTING_POINTS are later used to configure #
# the 'each nth point' key for some of the pgfplots axes.  #
# -------------------------------------------------------- #
PLOTTING_POINTS = 1000
NA_PLOT_SECTION = np.ceil(NA * (PLOT_STOP - PLOT_START)/(T_STOP - T_START))
DELTA_T     = (T_STOP - T_START) / NA
FA          = 1                    # analog base frequency #
FREQ_FACTOR = 11
FS          = FREQ_FACTOR * FA        # Sampling frequency #
TS          = 1/FS
TA_AXIS     = np.arange(T_START,T_STOP,DELTA_T)
TD_AXIS     = np.arange(T_START,T_STOP+TS,TS)
ND          = TD_AXIS.size
# -------------------------------------------------------- #
# Due to  numerical inaccuracies, the original  NA and the #
# TA_Axis length can deviate.  Make sure they do not.      #
# -------------------------------------------------------- #
NA          = TA_AXIS.size


# ########################################################################## #
# FUNCTIONS                                                                  #
# ########################################################################## #
def check_input(plot_start,
        plot_stop,
        t_start,
        t_stop,
        plotting_points,
        na_plot_section,
        delta_t,
        fa,
        freq_factor,
        fs,
        ts,
        ta_axis,
        td_axis,
        na,
        nd):

    assert isinstance(plot_start,(int,float))
    # Starting earlier than 0 has undesired results.
    assert plot_start >= 0
    assert isinstance(plot_stop,(int,float))
    assert plot_stop > t_start
    assert isinstance(t_start,(int,float))
    assert t_start >= 0
    assert isinstance(plotting_points ,(int))
    # ---------------------------------------------------- #
    # There should  be at least  as many analog  points in #
    # the  interval of  the  analog axis  which  is to  be #
    # plotted as there  are points which we say  are to be #
    # plotted.                                             #
    # ---------------------------------------------------- #
    assert plotting_points <= na_plot_section
    assert isinstance(t_stop ,(int,float))
    assert t_stop > t_start
    assert na >= 10 # This is an arbitrary lower limit.
    assert isinstance(fa,(int,float))
    assert fa > 0
    assert isinstance(freq_factor,(int,float))
    assert freq_factor > 0
    assert isinstance(fs,(int,float))
    assert fs > 0


def get_analog_signal(time_axis,analog_base_frequency):
    # Simple Sine ---------------------------------------------------------- #
    #return np.sin(2 * np.pi * analog_base_frequency * time_axis)

    # Trapezoidal Wave ----------------------------------------------------- #
    #http://www.till.com/articles/QuadTrapVCO/trapezoid.html
    signal = np.array(np.zeros(time_axis.size))
    for k in range(1,10):
        signal += 8/(np.pi * k)**2 * (
            (np.sin(k * np.pi/4) + np.sin(3 * k * np.pi/4)) * (
                np.sin(k*np.pi*analog_base_frequency*time_axis)))
    return signal


def find_closest_time_point(digital_time_point,analog_time_axis,delta_a):
    axis_with_single_dirac = np.zeros(analog_time_axis.size)

    j = 0
    for t in analog_time_axis:
        # For  the closest  possible analog time  point, set
        # dirac pulse to 1.
        if digital_time_point >= t - delta_a/2 and digital_time_point < t + delta_a/2:
            axis_with_single_dirac[j] = 1
            break
        j += 1
    return axis_with_single_dirac


def get_dirac_sequence(analog_time_axis,digital_time_axis,analog_sample_count,delta_a,pool):
    dirac_sequence = np.vstack([analog_time_axis,np.zeros(analog_sample_count)])

    results = sum(pool.starmap(
        find_closest_time_point,
        zip(
            digital_time_axis,
            repeat(analog_time_axis),
            repeat(delta_a)
        )
    ))
    dirac_sequence[1] = results

    return dirac_sequence


def strip_sampled_signal(signal_sampled_padded,dirac_sequence,digital_sample_count):
    j = 0
    k = 0
    signal_stripped = np.zeros(digital_sample_count)
    for value in signal_sampled_padded:
        if dirac_sequence[1,j] != 0:
            signal_stripped[k] = value
            k += 1
        j += 1
    return signal_stripped


def sample_signal(signal,dirac_sequence,digital_sample_count):
    signal_sampled_padded = np.multiply(signal,dirac_sequence[1])
    return strip_sampled_signal(signal_sampled_padded,dirac_sequence,digital_sample_count)


def reconstruct_at_analog_point(k,analog_time_point,digital_time_axis,signal_sampled,delta_a,Ts):
    signal_reconstruced_at_analog_point = 0
    j = 0
    for t in digital_time_axis:
        signal_reconstruced_at_analog_point += signal_sampled[j] * np.sinc((k * delta_a - j * Ts)/Ts)
        j += 1

    return signal_reconstruced_at_analog_point


def reconstruct_signal(analog_time_axis,digital_time_axis,signal_sampled,delta_a,Ts,analog_sample_count,pool):
    # Apparently these will be in the correct order. Cool.
    results = pool.starmap(
        reconstruct_at_analog_point,
        zip(
            range(0,analog_time_axis.size),
            analog_time_axis,
            repeat(digital_time_axis),
            repeat(signal_sampled),
            repeat(delta_a),
            repeat(Ts)
        )
    )

    return results


def plot_the_things(analog_time_axis,
        digital_time_axis,
        signal,signal_sampled,
        dirac_sequence,
        signal_reconstructed,
        plot_start,
        plot_stop):

    fig1 = plt.figure(figsize=[11.693,8.268])
    axes1 = fig1.add_subplot(221)
    axes2 = fig1.add_subplot(222)
    axes3 = fig1.add_subplot(223)
    axes4 = fig1.add_subplot(224)
    axes1.set_ylim([-1.1,1.1])
    axes2.set_ylim([-1.1,1.1])
    axes3.set_ylim([-1.1,1.1])
    axes4.set_ylim([-1.1,1.1])
    axes1.step(analog_time_axis,signal,color='red')
    axes1.stem(dirac_sequence[0],dirac_sequence[1])
    axes2.stem(digital_time_axis,signal_sampled)
    axes3.step(analog_time_axis,signal,color='r')
    axes3.scatter(digital_time_axis,signal_sampled)
    axes4.step(analog_time_axis,signal,color='r')
    axes4.step(analog_time_axis,signal_reconstructed)
    axes1.set_title('Analog Signal and Dirac Pulse Sequence',fontsize=10)
    axes2.set_title('Sampled Signal',fontsize=10)
    axes3.set_title('Sampled and Analog Signal Overlaid',fontsize=10)
    axes4.set_title('Original Signal and Reconstructed Signal',fontsize=10)
    axes1.set_xlim([plot_start,plot_stop])
    axes2.set_xlim([plot_start,plot_stop])
    axes3.set_xlim([plot_start,plot_stop])
    axes4.set_xlim([plot_start,plot_stop])
    fig1.suptitle(
        'Sampling and Reconstruction of an Analog Signal\nEvaluated Range: {:.2f} to {:.2f}'.format(T_START,T_STOP) +
        '\nNumber of Analog Points: {}'.format(NA) +
        '\nSampling Frequency={:.2f}*f_analog'.format(FREQ_FACTOR),
        fontsize=12)
    plt.show()
    #plt.tight_layout()
    #fig1.subplots_adjust(top=0.85,hspace=0.3)
    #fig1.savefig('test.pdf')

# -------------------------------------------------------- #
# Returns the minimum and maximum  values in a slice of an #
# array corresponding to a section of a global data set.   #
# -------------------------------------------------------- #
def get_y_bounds_in_slice(data,
        plotting_range_start,
        plotting_range_stop,
        global_range_start,
        global_range_stop,
        global_number_of_data_points):

    slice_start = int(plotting_range_start \
            / (global_range_stop - global_range_start) \
            * global_number_of_data_points)
    slice_stop  = int(plotting_range_stop \
            / (global_range_stop - global_range_start) \
            * global_number_of_data_points)

    y_lower_bound = np.min(data[slice_start:slice_stop])
    y_upper_bound = np.max(data[slice_start:slice_stop])
    return (y_lower_bound,y_upper_bound)


# -------------------------------------------------------- #
# Write  the   data  to  text  files   for  processing  in #
# pgfplots. Since TeX really  dislikes underscores in file #
# names, use camelCase instead.                            #
# -------------------------------------------------------- #
def write_to_files(analog_time_axis,
        digital_time_axis,
        signal,
        signal_sampled,
        dirac_sequence,
        signal_reconstructed,
        plot_start,
        plot_stop,
        plotting_points,
        na,
        na_plot_section,
        t_start,
        t_stop):

    analog_data        = np.vstack([analog_time_axis,signal])
    sampled_data       = np.vstack([digital_time_axis,signal_sampled])
    reconstructed_data = np.vstack([analog_time_axis,signal_reconstructed])
    error_data         = np.vstack([analog_time_axis,np.subtract(signal_reconstructed, signal)])

    np.savetxt('analogSignal.dat',       analog_data.transpose())
    np.savetxt('sampledSignal.dat',      sampled_data.transpose())
    np.savetxt('reconstructedSignal.dat',reconstructed_data.transpose())
    np.savetxt('errorSignal.dat',        error_data.transpose())

    # ---------------------------------------------------- #
    # We need to set upper  and lower boundaries for the y #
    # axis in  the error  plot because  otherwise pgfplots #
    # can throw  a '!  Dimension too  large' error  if the #
    # values get too  small (which is the case  for a high #
    # number of analog sample points).                     #
    # ---------------------------------------------------- #
    y_lower_bound,y_upper_bound = get_y_bounds_in_slice(error_data[1],
            plot_start,
            plot_stop,
            t_start,
            t_stop,
            na)

    nth = int(na_plot_section / plotting_points)
    with open('params.tex', 'w') as file:
        file.write('\def\plotStart{{{:.3f}}}\n'.format(plot_start))
        file.write('\def\plotStop{{{:.3f}}}\n'.format(plot_stop))
        file.write('\def\eachNth{{{:d}}}\n'.format(nth))
        file.write('\def\yLowerBound{{{:.8f}}}\n'.format(y_lower_bound * 1.1))
        file.write('\def\yUpperBound{{{:.8f}}}\n'.format(y_upper_bound * 1.1))


# ########################################################################## #
# MAIN SEQUENCE                                                              #
# ########################################################################## #
if __name__ == '__main__':
    check_input(PLOT_START,
            PLOT_STOP,
            T_START,
            T_STOP,
            PLOTTING_POINTS,
            NA_PLOT_SECTION,
            DELTA_T,FA,
            FREQ_FACTOR,
            FS,
            TS,
            TA_AXIS,
            TD_AXIS,
            NA,
            ND)

    pool = mp.Pool()

    # Time Domain ---------------------------------------------------------- #
    dirac_sequence       = get_dirac_sequence(TA_AXIS,TD_AXIS,NA,DELTA_T,pool)
    signal               = get_analog_signal(TA_AXIS,FA)
    signal_sampled       = sample_signal(signal,dirac_sequence,ND)
    signal_reconstructed = reconstruct_signal(TA_AXIS,TD_AXIS,signal_sampled,DELTA_T,TS,NA,pool)

    # Frequency Domain ----------------------------------------------------- #
    # DFT of sampled signal
    # IDFT of DFT of sampled signal
    # DFT of analog signal
    # sampling of DFT of analog signal
    # IDFT of samples of DFT of analog signal

    # Convolution ---------------------------------------------------------- #

    write_to_files(TA_AXIS,
            TD_AXIS,
            signal,
            signal_sampled,
            dirac_sequence,
            signal_reconstructed,
            PLOT_START,
            PLOT_STOP,
            PLOTTING_POINTS,
            NA,
            NA_PLOT_SECTION,
            T_START,
            T_STOP)
    #plot_the_things(TA_AXIS,TD_AXIS,
    #        signal,
    #        signal_sampled,
    #        dirac_sequence,
    #        signal_reconstructed,
    #        PLOT_START,PLOT_STOP)
