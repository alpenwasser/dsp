#!/usr/bin/env python3

from itertools import repeat
import shutil as shutil
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import sys as sys
import os as os
import argparse as argp
import re as re

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
# FUNCTIONS                                                                  #
# ########################################################################## #
def check_input(
        plot_start, plot_stop,
        t_start, t_stop,
        plotting_points,
        na_plot_section,
        delta_t,
        fa,
        freq_factor,
        fs, ts,
        ta_axis, td_axis,
        na, nd):

    # ---------------------------------------------------- #
    # Starting earlier than 0 has undesired results, so we #
    # don't.                                               #
    # ---------------------------------------------------- #
    assert plot_start >= 0
    assert plot_stop > t_start
    assert t_start >= 0
    # ---------------------------------------------------- #
    # There should  be at least  as many analog  points in #
    # the  interval of  the  analog axis  which  is to  be #
    # plotted as there  are points which we say  are to be #
    # plotted.                                             #
    # ---------------------------------------------------- #
    assert plotting_points <= na_plot_section
    assert t_stop > t_start
    assert na >= 10 # This is an arbitrary lower limit.
    assert fa > 0
    assert freq_factor > 0
    assert fs > 0


def display_config(
        plot_start, plot_stop,
        t_start, t_stop,
        plotting_points,
        na_plot_section,
        delta_t,
        fa,
        freq_factor,
        fs, ts,
        display, store, datadir,
        na, nd):

    print('Processing Signal with the following parameters')
    print('----------------------------------------------------')
    print('Plot Range                      {: >8.2f} to {: >8.2f}'.format(plot_start, plot_stop))
    print('Signal Range                    {: >8.2f} to {: >8.2f}'.format(t_start, t_stop))
    print('Analog Points in Signal Range   {: >8.0e}'.format(na))
    print('Analog Points in Plot Range     {: >8.0e}'.format(na_plot_section))
    print('Points to Plot in TeX           {: >8.0e}'.format(plotting_points))
    print('Digital Samples                 {: >8.0e}'.format(nd))
    print('Analog Base Frequency           {: >8.2f} Hz'.format(fa))
    print('Sampling Factor                 {: >8.2f}'.format(freq_factor))
    print('Sampling Frequency              {: >8.2f} Hz'.format(fs))

    if store:
        print('Export Data                         True')
        print('Data Dir Relative to Working Dir    {}'.format(datadir))
    else:
        print('Export Data                        False')

    if display:
        print('Display Data in Matplotlib          True')
    else:
        print('Display Data in Matplotlib         False')


def get_analog_signal(time_axis, analog_base_frequency, waveform):

    # ---------------------------------------------------- #
    # Returns  the y  values for  a given  time axis  at a #
    # given  base frequeny.   The waveform  argument is  a #
    # string  specifying which  sort  of signal  is to  be #
    # returned.  The default signal is a sine wave.        #
    # ---------------------------------------------------- #

    if waveform == 'trapezoid':
        # Trapezoidal Wave ------------------------------- #
        #http://www.till.com/articles/QuadTrapVCO/trapezoid.html
        signal = np.array(np.zeros(time_axis.size))
        for k in range(1,10):
            signal += 8/(np.pi * k)**2 * (
                (np.sin(k * np.pi/4) + np.sin(3 * k * np.pi/4)) * (
                    np.sin(k*np.pi*analog_base_frequency*time_axis)))
        return signal
    else:
        # Sine ------------------------------------------- #
        return np.sin(2 * np.pi * analog_base_frequency * time_axis)


def find_closest_time_point(digital_time_point,analog_time_axis,delta_a):

    axis_with_single_dirac = np.zeros(analog_time_axis.size)

    j = 0   # analog point index
    for t in analog_time_axis:
        # ------------------------------------------------ #
        # For the closest possible  analog time point, set #
        # dirac pulse to 1.                                #
        # ------------------------------------------------ #
        if digital_time_point >= t - delta_a/2 \
                and digital_time_point < t + delta_a/2:
            axis_with_single_dirac[j] = 1
            break
        j += 1
    return axis_with_single_dirac


def get_dirac_sequence(
        analog_time_axis,digital_time_axis,analog_sample_count,delta_a,pool):

    dirac_sequence = np.vstack(
            [analog_time_axis,np.zeros(analog_sample_count)])

    # ---------------------------------------------------- #
    # Each process started by  starmap will return a dirac #
    # sequence with  one single impulse in  it. To get the #
    # complete  axis with  all  the needed  pulses for  an #
    # entire sequence, we simply sum  up all the axes with #
    # a single pulse in them.                              #
    # ---------------------------------------------------- #
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


def strip_sampled_signal(
        signal_sampled_padded,dirac_sequence,digital_sample_count):

    j = 0 # analog point index (padded)
    k = 0 # digital sample index
    signal_stripped = np.zeros(digital_sample_count)
    for value in signal_sampled_padded:
        if dirac_sequence[1,j] != 0:
            signal_stripped[k] = value
            k += 1
        j += 1
    return signal_stripped


def sample_signal(signal,dirac_sequence,digital_sample_count):
    signal_sampled_padded = np.multiply(signal,dirac_sequence[1])
    return strip_sampled_signal(
            signal_sampled_padded,dirac_sequence,digital_sample_count)


def reconstruct_at_analog_point(
        k, analog_time_point, digital_time_axis, signal_sampled, delta_a, Ts):

    # ---------------------------------------------------- #
    # Calculates the  y value of the  reconstructed signal #
    # at one specific time point.                          #
    # ---------------------------------------------------- #
    signal_reconstruced_at_analog_point = 0
    j = 0
    for t in digital_time_axis:
        signal_reconstruced_at_analog_point += (
                signal_sampled[j] * np.sinc((k * delta_a - j * Ts)/Ts))
        j += 1

    return signal_reconstruced_at_analog_point


def reconstruct_signal(
        analog_time_axis, digital_time_axis,
        signal_sampled,
        delta_a,
        Ts,
        analog_sample_count,
        pool):

    # ---------------------------------------------------- #
    # results    will   contain    the    y   values    of #
    # the    reconstructed    signal   (each    y    value #
    # having    been   calculated    by   the    call   to #
    # reconstruct_at_analog_point). Apparently,    starmap #
    # returns  them   in  the  correct  order,   so  there #
    # is  no  need  to  track which  y  value  belongs  to #
    # which  time  point  (which  could  be  done  in  the #
    # reconstruct_at_analog_point  routine,   if  it  ever #
    # becomes necessary).                                  #
    # ---------------------------------------------------- #
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


def plot_the_things(
        analog_time_axis, digital_time_axis,
        signal, signal_sampled,
        dirac_sequence,
        signal_reconstructed,
        plot_start, plot_stop,
        t_start, t_stop,
        freq_factor,
        na):

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
        'Sampling and Reconstruction of an Analog Signal\nEvaluated Range: {:.2f} to {:.2f}'.format(t_start,t_stop) +
        '\nNumber of Analog Points: {}'.format(na) +
        '\nSampling Frequency={:.2f}*f_analog'.format(freq_factor),
        fontsize=12)
    plt.show()


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


def generate_tex_file(
        tex_file,
        data_dir,
        analog_file_path,sampled_file_path,reconstr_file_path,error_file_path,
        error_data, plot_start, plot_stop, t_start, t_stop,
        na, na_plot_section, plotting_points):

    tex_template = os.path.join('TeX','template.tex')
    tex_file_path = os.path.join('TeX', tex_file)
    param_file = os.path.join('TeX','params.tex')

    # ---------------------------------------------------- #
    # We create a new TeX  file with the parameters in its #
    # filename. This avoids  having to manually  name each #
    # plot pdf file.                                       #
    # ---------------------------------------------------- #
    shutil.copy(tex_template,tex_file_path)

    # Relative file paths from TeX file to data files
    tex_analog_file_path   = os.path.join('..', analog_file_path)
    tex_sampled_file_path  = os.path.join('..', sampled_file_path)
    tex_reconstr_file_path = os.path.join('..', reconstr_file_path)
    tex_error_file_path    = os.path.join('..', error_file_path)


    # ---------------------------------------------------- #
    # We need to set upper  and lower boundaries for the y #
    # axis in  the error  plot because  otherwise pgfplots #
    # can throw  a '!  Dimension too  large' error  if the #
    # values get too  small (which is the case  for a high #
    # number of analog sample points).                     #
    # ---------------------------------------------------- #
    y_lower_bound,y_upper_bound = get_y_bounds_in_slice(
            error_data[1],
            plot_start, plot_stop,
            t_start, t_stop,
            na)

    nth = int(na_plot_section / plotting_points)
    with open(param_file, 'w') as file:
        file.write('\def\plotStart{{{:.3f}}}\n'.format(plot_start))
        file.write('\def\plotStop{{{:.3f}}}\n'.format(plot_stop))
        file.write('\def\eachNth{{{:d}}}\n'.format(nth))
        file.write('\def\yLowerBound{{{:.8f}}}\n'.format(y_lower_bound * 1.1))
        file.write('\def\yUpperBound{{{:.8f}}}\n'.format(y_upper_bound * 1.1))
        file.write('\def\\analogFilePath{{{}}}\n'.format(tex_analog_file_path))
        file.write('\def\sampledFilePath{{{}}}\n'.format(tex_sampled_file_path))
        file.write('\def\\reconstrFilePath{{{}}}\n'.format(tex_reconstr_file_path))
        file.write('\def\errorFilePath{{{}}}\n'.format(tex_error_file_path))


# -------------------------------------------------------- #
# Write  the   data  to  text  files   for  processing  in #
# pgfplots. Since TeX really  dislikes underscores in file #
# names, use camelCase instead.                            #
# -------------------------------------------------------- #
def write_to_files(
        analog_time_axis, digital_time_axis,
        signal, signal_sampled,
        dirac_sequence,
        signal_reconstructed,
        plot_start, plot_stop,
        plotting_points, na, na_plot_section,
        t_start, t_stop,
        waveform,
        freq_factor,
        data_dir):

    analog_data        = np.vstack([analog_time_axis,signal])
    sampled_data       = np.vstack([digital_time_axis,signal_sampled])
    reconstructed_data = np.vstack([analog_time_axis,signal_reconstructed])
    error_data         = np.vstack([analog_time_axis,np.subtract(signal_reconstructed, signal)])

    os.makedirs(data_dir, exist_ok=True)

    # ---------------------------------------------------- #
    # Create  a   prefix  corresponding  to   the  current #
    # configuration. Some rounding is involved.            #
    # ---------------------------------------------------- #
    prefix = '{}--{:.0e}-anPts--fs-{:.1f}-fa--range-{:.0f}--zoom-{:.0f}'.format(
            waveform,
            na,
            freq_factor,
            t_stop-t_start,
            plot_stop-plot_start
        )

    tex_file = prefix + '.tex'
    analog_file = prefix + '--analogSignal.dat'
    sampled_file = prefix + '--sampledSignal.dat'
    reconstr_file = prefix + '--reconstructedSignal.dat'
    error_file = prefix + '--errorSignal.dat'

    analog_file_path = os.path.join(data_dir,analog_file)
    sampled_file_path = os.path.join(data_dir,sampled_file)
    reconstr_file_path = os.path.join(data_dir,reconstr_file)
    error_file_path = os.path.join(data_dir,error_file)

    np.savetxt(analog_file_path,   analog_data.transpose())
    np.savetxt(sampled_file_path,  sampled_data.transpose())
    np.savetxt(reconstr_file_path, reconstructed_data.transpose())
    np.savetxt(error_file_path,    error_data.transpose())

    generate_tex_file(
        tex_file,
        data_dir,
        analog_file_path,sampled_file_path,reconstr_file_path,error_file_path,
        error_data, plot_start, plot_stop, t_start, t_stop,
        na, na_plot_section, plotting_points)


# ########################################################################## #
# MAIN SEQUENCE                                                              #
# ########################################################################## #
def main(argv):

    # Parse Input Arguments. New Input Arguments Should Be Defined Here ---- #
    parser = argp.ArgumentParser()
    parser.add_argument("-s",
            "--plotstart",
            type=float,
            default = 9 * np.pi,
            help = "the starting point for the plot window")
    parser.add_argument("-e",
            "--plotstop",
            type=float,
            default = 11 * np.pi,
            help = "the end point for the plot window")
    parser.add_argument("-t",
            "--totalpoints",
            type=float,
            default = 1e5,
            help = "the total number of points on the analog axis")
    parser.add_argument("-p",
            "--plottingpoints",
            type=float,
            default = 1000,
            help = "the number of points to be plotted")
    parser.add_argument("-a",
            "--analogfreq",
            type=float,
            default = 1,
            help = "the analog base frequency")
    parser.add_argument("-f",
            "--samplingfactor",
            type=float,
            default = 3,
            help = "factor for calculating sampling frequency from base frequency")
    parser.add_argument("-w",
            "--waveform",
            type=str,
            default = 'sine',
            choices = ['sine', 'trapezoid'],
            help = "the waveform for the analog signal")
    parser.add_argument("-d",
            "--display",
            action = 'store_true',
            help = "display the waves in a matplotlib window")
    parser.add_argument("-o",
            "--store",
            action = 'store_true',
            help = "Output results to data files. Automatically True if -i specified.")
    parser.add_argument("-i",
            "--datadir",
            type=str,
            default = 'data',
            help = "The directory to store data files. Sets -o to True if specified.")
    args = parser.parse_args()

    # Copy Arguments Into Variables for Further Use ------------------------ #
    if args.plotstart:
        plot_start = args.plotstart
    if args.plotstop:
        plot_stop = args.plotstop
    if args.totalpoints:
        na = args.totalpoints
    if args.plottingpoints:
        plotting_points = args.plottingpoints
    if args.analogfreq:
        fa = args.analogfreq
    if args.samplingfactor:
        freq_factor = args.samplingfactor
    if args.waveform:
        waveform = args.waveform
    if args.datadir:
        datadir = args.datadir
        store = True

    # Open a matplotlib Window if True ------------------------------------- #
    if args.display:
        display = True
    else:
        display = False

    # Store the Results to Files if True ----------------------------------- #
    if args.store:
        store = True
    else:
        store = False

    # Determine Parameters from Input Arguments ---------------------------- #
    t_start         =  0
    t_stop          = plot_stop + (plot_start - t_start)
    na_plot_section = np.ceil(na * (plot_stop - plot_start)/(t_stop - t_start))
    delta_t         = (t_stop - t_start) / na
    fs              = freq_factor * fa                  # sampling frequency #
    ts              = 1/fs
    ta_axis         = np.arange(t_start,t_stop,delta_t)
    td_axis         = np.arange(t_start,t_stop+ts,ts)
    nd              = td_axis.size
    na              = ta_axis.size

    # Display  Configuration ----------------------------------------------- #
    display_config(
            plot_start, plot_stop,
            t_start, t_stop,
            plotting_points,
            na_plot_section,
            delta_t,
            fa, freq_factor, fs,
            ts,
            display, store, datadir,
            na, nd)

    # Create Process Pool -------------------------------------------------- #
    pool = mp.Pool()

    # Verify Some Additional Constraints ----------------------------------- #
    check_input(
            plot_start, plot_stop,
            t_start, t_stop,
            plotting_points,
            na_plot_section,
            delta_t,
            fa, freq_factor, fs,
            ts,
            ta_axis, td_axis,
            na, nd)

    # Time Domain ---------------------------------------------------------- #
    dirac_sequence = get_dirac_sequence(
            ta_axis, td_axis,
            na,
            delta_t,
            pool)

    signal = get_analog_signal(ta_axis, fa, waveform)

    signal_sampled = sample_signal(signal, dirac_sequence, nd)

    signal_reconstructed = reconstruct_signal(
            ta_axis, td_axis,
            signal_sampled,
            delta_t,
            ts,
            na,
            pool)

    # Frequency Domain ----------------------------------------------------- #
    # DFT of sampled signal
    # IDFT of DFT of sampled signal
    # DFT of analog signal
    # sampling of DFT of analog signal
    # IDFT of samples of DFT of analog signal

    # Noise ---------------------------------------------------------------- #

    # Convolution ---------------------------------------------------------- #

    # Power ---------------------------------------------------------------- #

    # Filtering ------------------------------------------------------------ #

    # Output (Optional, Disabled by Default) ------------------------------- #
    if store:
        write_to_files(
                ta_axis, td_axis,
                signal, signal_sampled,
                dirac_sequence,
                signal_reconstructed,
                plot_start, plot_stop,
                plotting_points, na, na_plot_section,
                t_start, t_stop,
                waveform,
                freq_factor,
                datadir)

    if display:
        plot_the_things(
                ta_axis, td_axis,
                signal, signal_sampled,
                dirac_sequence,
                signal_reconstructed,
                plot_start, plot_stop,
                t_start, t_stop,
                freq_factor,
                na)


# ########################################################################## #
# ENTRY POINT                                                                #
# ########################################################################## #

if __name__ == '__main__':
    main(sys.argv[1:])
