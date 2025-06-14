def split_data_em(y: np.ndarray,
                  labels: np.ndarray,
                  test_size: int = 4):
    """
    Split data into balanced train/test sets and oversample minority in train.

    Parameters
    ----------
    y : np.ndarray
        Feature array of shape (n_samples, ...).
    labels : np.ndarray
        Integer labels array of shape (n_samples,), values in {0,1}.
    test_size : int
        Total number of test samples (must be even), default 4.

    Returns
    -------
    y_train : np.ndarray
        Training features.
    label_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test features.
    label_test : np.ndarray
        Test labels.
    train_indices : np.ndarray
        Indices used for training.
    test_indices : np.ndarray
        Indices used for testing.
    """
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    test_idx0 = np.random.choice(idx0, size=test_size // 2, replace=False)
    test_idx1 = np.random.choice(idx1, size=test_size // 2, replace=False)
    test_indices = np.concatenate([test_idx0, test_idx1])

    remaining_idx0 = np.setdiff1d(idx0, test_idx0)
    remaining_idx1 = np.setdiff1d(idx1, test_idx1)

    num0, num1 = len(remaining_idx0), len(remaining_idx1)
    if num0 > num1:
        extra_idx1 = np.random.choice(remaining_idx1, size=num0 - num1, replace=True)
        train_indices = np.concatenate([remaining_idx0, remaining_idx1, extra_idx1])
    elif num1 > num0:
        extra_idx0 = np.random.choice(remaining_idx0, size=num1 - num0, replace=True)
        train_indices = np.concatenate([remaining_idx0, remaining_idx1, extra_idx0])
    else:
        train_indices = np.concatenate([remaining_idx0, remaining_idx1])

    np.random.shuffle(train_indices)

    y_train, y_test = y[train_indices], y[test_indices]
    label_train, label_test = labels[train_indices], labels[test_indices]

    print(f"Train set size for class 0: {np.sum(label_train == 0)}")
    print(f"Train set size for class 1: {np.sum(label_train == 1)}")
    print(f"Test set size for class 0: {np.sum(label_test == 0)}")
    print(f"Test set size for class 1: {np.sum(label_test == 1)}")

    return y_train, label_train, y_test, label_test, train_indices, test_indices


def balance_classes(y: np.ndarray,
                    labels: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Balance classes by dropping one random sample from the majority class.

    Parameters
    ----------
    y : np.ndarray
        Feature array of shape (n_samples, ...).
    labels : np.ndarray
        Integer labels array of shape (n_samples,).

    Returns
    -------
    balanced_y : np.ndarray
        Features after dropping one sample.
    balanced_labels : np.ndarray
        Labels after dropping one sample.
    """
    class_0_indices = np.where(labels == 0)[0]
    class_1_indices = np.where(labels == 1)[0]
    if len(class_0_indices) > len(class_1_indices):
        excess = class_0_indices
    else:
        excess = class_1_indices
    drop_index = np.random.choice(excess, size=1, replace=False)
    balanced_y = np.delete(y, drop_index, axis=0)
    balanced_labels = np.delete(labels, drop_index, axis=0)
    return balanced_y, balanced_labels


def print_cnn_summary(cnn_variables: dict):
    """
    Print the shape of each parameter tensor in a Flax CNN model.

    Parameters
    ----------
    cnn_variables : dict
        Flax variables dict containing 'params'.
    """
    print("CNN model parameters:")
    for layer, params in cnn_variables['params'].items():
        print(f"Layer: {layer}")
        for param_name, value in params.items():
            print(f"  {param_name}: {value.shape}")


def plot_conv_weights(cnn_variables: dict):
    """
    Plot the convolutional kernels of the first two Conv layers.

    Parameters
    ----------
    cnn_variables : dict
        Flax variables dict containing 'params'.
    """
    params = cnn_variables['params']

    # Plot Conv_0 kernels
    conv0 = params['Conv_0']['kernel']
    n0 = conv0.shape[-1]
    plt.figure(figsize=(20, 4))
    for i in range(n0):
        plt.subplot(1, n0, i+1)
        plt.plot(conv0[:, 0, i])
        plt.title(f"Conv1 Filter {i}")
        plt.xlabel("Kernel Index")
        plt.ylabel("Weight")
    plt.suptitle("Initial Conv1 Weights")
    plt.tight_layout()
    plt.show()

    # Plot Conv_1 kernels
    conv1 = params['Conv_1']['kernel']
    n1 = conv1.shape[-1]
    fig, axs = plt.subplots(n1, 1, figsize=(10, n1*2))
    for i in range(n1):
        for j in range(conv1.shape[1]):
            axs[i].plot(conv1[:, j, i], label=f"Input Ch {j}")
        axs[i].set_title(f"Conv2 Filter {i}")
        axs[i].set_xlabel("Kernel Index")
        axs[i].set_ylabel("Weight")
        axs[i].legend()
    plt.suptitle("Initial Conv2 Weights", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_intermediate_activations(cnn_variables: dict,
                                  cnn_apply_fn: callable,
                                  dummy_input: jnp.ndarray):
    """
    Plot activations from the first two convolutional layers on a dummy input.

    Parameters
    ----------
    cnn_variables : dict
        Flax variables dict containing 'params' and 'batch_stats'.
    cnn_apply_fn : callable
        Flax model apply function.
    dummy_input : jnp.ndarray
        Single input example, shape matching model input (1, T, 1).
    """
    outputs, intermediates = cnn_model.apply(
        cnn_variables, dummy_input, train=False, mutable=['intermediates']
    )

    # Conv1 activations
    conv1_act = intermediates['intermediates']['conv1'][0][0]
    c1 = conv1_act.shape[-1]
    plt.figure(figsize=(20, 3*c1))
    for i in range(c1):
        plt.subplot(c1, 1, i+1)
        plt.plot(conv1_act[:, i])
        plt.title(f"Conv1 Activation - Channel {i}")
        plt.xlabel("Time Step")
        plt.ylabel("Activation")
    plt.tight_layout()
    plt.show()

    # Conv2 activations
    conv2_act = intermediates['intermediates']['conv2'][0][0]
    c2 = conv2_act.shape[-1]
    plt.figure(figsize=(20, 3*c2))
    for i in range(c2):
        plt.subplot(c2, 1, i+1)
        plt.plot(conv2_act[:, i])
        plt.title(f"Conv2 Activation - Channel {i}")
        plt.xlabel("Time Step")
        plt.ylabel("Activation")
    plt.tight_layout()
    plt.show()


def multitaper_spectrogram(signal: np.ndarray,
                           fs: float,
                           frequency_range,
                           time_bandwidth,
                           num_tapers,
                           window_params,
                           min_nfft: int,
                           detrend_opt: str,
                           multiprocess: bool,
                           n_jobs: int,
                           weighting: str,
                           plot_on: bool,
                           return_fig: bool,
                           clim_scale: bool,
                           verbose: bool,
                           xyflip: bool):
    """
    Simplified multitaper spectrogram: compute one-sided FFT magnitude.

    Parameters
    ----------
    signal : np.ndarray
        1D time-series signal.
    fs : float
        Sampling frequency.
    frequency_range : sequence
        Frequency bounds [f_min, f_max].
    time_bandwidth : float
        Time-bandwidth product (ignored here).
    num_tapers : int
        Number of tapers (ignored here).
    window_params : sequence
        Window length params (ignored here).
    min_nfft : int
        Minimum FFT length (ignored here).
    detrend_opt : str
        Detrending option (ignored here).
    multiprocess : bool
        Whether to parallelize (ignored here).
    n_jobs : int
        Number of jobs if multiprocessing (ignored here).
    weighting : str
        Spectral weighting (ignored here).
    plot_on : bool
        Whether to plot (ignored here).
    return_fig : bool
        Whether to return figure handle (ignored here).
    clim_scale : bool
        Color scale option (ignored here).
    verbose : bool
        Verbosity flag (ignored here).
    xyflip : bool
        Whether to flip axes (ignored here).

    Returns
    -------
    spectrum : np.ndarray
        One-sided FFT magnitude of length len(signal)//2.
    None
        Placeholder for compatibility.
    f : np.ndarray
        Frequencies corresponding to spectrum, from 0 to fs/2.
    """
    fft_complex = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_complex)
    num_freq = len(signal) // 2
    spectrum = fft_magnitude[:num_freq]
    f = np.linspace(0, fs/2, num_freq)
    return spectrum, None, f


def compute_particle_spectrum(traj: jnp.ndarray,
                              T_fixed: int) -> jnp.ndarray:
    """
    Compute an FFT-based spectrogram for a single trajectory, padding or truncating
    to length T_fixed, then apply a frequency mask for CNN input.

    Parameters
    ----------
    traj : jnp.ndarray
        1D trajectory array of shape (T,).
    T_fixed : int
        Fixed length to reshape trajectory to.

    Returns
    -------
    reduced_spectrum : jnp.ndarray
        Array of shape (1, reduced_freq, 1) matching CNN input format.
    """
    current_T = traj.shape[0]
    if current_T >= T_fixed:
        traj_fixed = traj[:T_fixed]
    else:
        traj_fixed = jnp.pad(traj, (0, T_fixed - current_T))

    data = traj_fixed.reshape(1, T_fixed, 1)
    fft_complex = jnp.fft.fft(data, axis=1)
    fft_magnitude = jnp.abs(fft_complex)
    num_freq = T_fixed // 2
    spectrum = fft_magnitude[:, :num_freq, :]
    f = jnp.linspace(0, 0.5, num_freq)
    mask = f <= 0.035
    reduced_spectrum = spectrum[:, mask, :]
    return reduced_spectrum


def compute_spectra_per_particle(data: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Compute FFT-based spectrograms for each particle in each trial using multitaper stub.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (num_trials, num_time, num_particles) containing trajectories.

    Returns
    -------
    all_spectra : np.ndarray
        Array of shape (num_trials, num_freq, num_particles) of spectra.
    f : np.ndarray
        Frequency vector corresponding to rows of all_spectra.
    """
    num_trials, num_time, num_particles = data.shape
    example_signal = data[0, :, 0]
    example_spectrum, _, f = multitaper_spectrogram(
        example_signal, fs=1, frequency_range=[0, 0.5],
        time_bandwidth=2, num_tapers=3, window_params=[360, 360],
        min_nfft=0, detrend_opt='constant', multiprocess=False, n_jobs=3,
        weighting='unity', plot_on=False, return_fig=False,
        clim_scale=False, verbose=True, xyflip=False
    )
    num_freq = example_spectrum.shape[0]
    all_spectra = np.zeros((num_trials, num_freq, num_particles))
    for t in range(num_trials):
        for p in range(num_particles):
            sig = data[t, :, p]
            spec, _, _ = multitaper_spectrogram(
                sig, fs=1, frequency_range=[0, 0.5],
                time_bandwidth=2, num_tapers=3, window_params=[360, 360],
                min_nfft=0, detrend_opt='constant', multiprocess=False, n_jobs=3,
                weighting='unity', plot_on=False, return_fig=False,
                clim_scale=False, verbose=False, xyflip=False
            )
            all_spectra[t, :, p] = np.squeeze(spec)
    return all_spectra, f

def multitaper_spectrogram(data, fs, frequency_range=None, time_bandwidth=5, num_tapers=None, window_params=None,
                           min_nfft=0, detrend_opt='linear', multiprocess=False, n_jobs=None, weighting='unity',
                           plot_on=True, return_fig=False, clim_scale=True, verbose=True, xyflip=False, ax=None):
    """ Compute multitaper spectrogram of timeseries data
    Usage:
    mt_spectrogram, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range=None, time_bandwidth=5,
                                                            num_tapers=None, window_params=None, min_nfft=0,
                                                            detrend_opt='linear', multiprocess=False, cpus=False,
                                                            weighting='unity', plot_on=True, return_fig=False,
                                                            clim_scale=True, verbose=True, xyflip=False):
        Arguments:
                data (1d np.array): time series data -- required
                fs (float): sampling frequency in Hz  -- required
                frequency_range (list): 1x2 list - [<min frequency>, <max frequency>] (default: [0 nyquist])
                time_bandwidth (float): time-half bandwidth product (window duration*half bandwidth of main lobe)
                                        (default: 5 Hz*s)
                num_tapers (int): number of DPSS tapers to use (default: [will be computed
                                  as floor(2*time_bandwidth - 1)])
                window_params (list): 1x2 list - [window size (seconds), step size (seconds)] (default: [5 1])
                detrend_opt (string): detrend data window ('linear' (default), 'constant', 'off')
                                      (Default: 'linear')
                min_nfft (int): minimum allowable NFFT size, adds zero padding for interpolation (closest 2^x)
                                (default: 0)
                multiprocess (bool): Use multiprocessing to compute multitaper spectrogram (default: False)
                n_jobs (int): Number of cpus to use if multiprocess = True (default: False). Note: if default is left
                            as None and multiprocess = True, the number of cpus used for multiprocessing will be
                            all available - 1.
                weighting (str): weighting of tapers ('unity' (default), 'eigen', 'adapt');
                plot_on (bool): plot results (default: True)
                return_fig (bool): return plotted spectrogram (default: False)
                clim_scale (bool): automatically scale the colormap on the plotted spectrogram (default: True)
                verbose (bool): display spectrogram properties (default: True)
                xyflip (bool): transpose the mt_spectrogram output (default: False)
                ax (axes): a matplotlib axes to plot the spectrogram on (default: None)
        Returns:
                mt_spectrogram (TxF np array): spectral power matrix
                stimes (1xT np array): timepoints (s) in mt_spectrogram
                sfreqs (1xF np array)L frequency values (Hz) in mt_spectrogram

        Example:
        In this example we create some chirp data and run the multitaper spectrogram on it.
            import numpy as np  # import numpy
            from scipy.signal import chirp  # import chirp generation function
            # Set spectrogram params
            fs = 200  # Sampling Frequency
            frequency_range = [0, 25]  # Limit frequencies from 0 to 25 Hz
            time_bandwidth = 3  # Set time-half bandwidth
            num_tapers = 5  # Set number of tapers (optimal is time_bandwidth*2 - 1)
            window_params = [4, 1]  # Window size is 4s with step size of 1s
            min_nfft = 0  # No minimum nfft
            detrend_opt = 'constant'  # detrend each window by subtracting the average
            multiprocess = True  # use multiprocessing
            cpus = 3  # use 3 cores in multiprocessing
            weighting = 'unity'  # weight each taper at 1
            plot_on = True  # plot spectrogram
            return_fig = False  # do not return plotted spectrogram
            clim_scale = False # don't auto-scale the colormap
            verbose = True  # print extra info
            xyflip = False  # do not transpose spect output matrix

            # Generate sample chirp data
            t = np.arange(1/fs, 600, 1/fs)  # Create 10 min time array from 1/fs to 600 stepping by 1/fs
            f_start = 1  # Set chirp freq range min (Hz)
            f_end = 20  # Set chirp freq range max (Hz)
            data = chirp(t, f_start, t[-1], f_end, 'logarithmic')
            # Compute the multitaper spectrogram
            spect, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range, time_bandwidth, num_tapers,
                                                           window_params, min_nfft, detrend_opt, multiprocess,
                                                           cpus, weighting, plot_on, return_fig, clim_scale,
                                                           verbose, xyflip):

        This code is companion to the paper:
        "Sleep Neurophysiological Dynamics Through the Lens of Multitaper Spectral Analysis"
           Michael J. Prerau, Ritchie E. Brown, Matt T. Bianchi, Jeffrey M. Ellenbogen, Patrick L. Purdon
           December 7, 2016 : 60-92
           DOI: 10.1152/physiol.00062.2015
         which should be cited for academic use of this code.

         A full tutorial on the multitaper spectrogram can be found at: # https://www.sleepEEG.org/multitaper

        Copyright 2021 Michael J. Prerau Laboratory. - https://www.sleepEEG.org
        Authors: Michael J. Prerau, Ph.D., Thomas Possidente, Mingjian He

  __________________________________________________________________________________________________________________
    """

    #  Process user input
    [data, fs, frequency_range, time_bandwidth, num_tapers,
     winsize_samples, winstep_samples, window_start,
     num_windows, nfft, detrend_opt, plot_on, verbose] = process_input(data, fs, frequency_range, time_bandwidth,
                                                                       num_tapers, window_params, min_nfft,
                                                                       detrend_opt, plot_on, verbose)

    # Set up spectrogram parameters
    [window_idxs, stimes, sfreqs, freq_inds] = process_spectrogram_params(fs, nfft, frequency_range, window_start,
                                                                          winsize_samples)
    # Display spectrogram parameters
    if verbose:
        display_spectrogram_props(fs, time_bandwidth, num_tapers, [winsize_samples, winstep_samples], frequency_range,
                                  nfft, detrend_opt)

    # Split data into segments and preallocate
    data_segments = data[window_idxs]

    # COMPUTE THE MULTITAPER SPECTROGRAM
    #     STEP 1: Compute DPSS tapers based on desired spectral properties
    #     STEP 2: Multiply the data segment by the DPSS Tapers
    #     STEP 3: Compute the spectrum for each tapered segment
    #     STEP 4: Take the mean of the tapered spectra

    # Compute DPSS tapers (STEP 1)
    dpss_tapers, dpss_eigen = dpss(winsize_samples, time_bandwidth, num_tapers, return_ratios=True)
    dpss_eigen = np.reshape(dpss_eigen, (num_tapers, 1))

    # pre-compute weights
    if weighting == 'eigen':
        wt = dpss_eigen / num_tapers
    elif weighting == 'unity':
        wt = np.ones(num_tapers) / num_tapers
        wt = np.reshape(wt, (num_tapers, 1))  # reshape as column vector
    else:
        wt = 0

    tic = timeit.default_timer()  # start timer

    # Set up calc_mts_segment() input arguments
    mts_params = (dpss_tapers, nfft, freq_inds, detrend_opt, num_tapers, dpss_eigen, weighting, wt)

    if multiprocess:  # use multiprocessing
        n_jobs = max(cpu_count() - 1, 1) if n_jobs is None else n_jobs
        mt_spectrogram = np.vstack(Parallel(n_jobs=n_jobs)(delayed(calc_mts_segment)(
            data_segments[num_window, :], *mts_params) for num_window in range(num_windows)))

    else:  # if no multiprocessing, compute normally
        mt_spectrogram = np.apply_along_axis(calc_mts_segment, 1, data_segments, *mts_params)

    # Compute one-sided PSD spectrum
    mt_spectrogram = mt_spectrogram.T
    dc_select = np.where(sfreqs == 0)[0]
    nyquist_select = np.where(sfreqs == fs/2)[0]
    select = np.setdiff1d(np.arange(0, len(sfreqs)), np.concatenate((dc_select, nyquist_select)))

    mt_spectrogram = np.vstack([mt_spectrogram[dc_select, :], 2*mt_spectrogram[select, :],
                               mt_spectrogram[nyquist_select, :]]) / fs

    # Flip if requested
    if xyflip:
        mt_spectrogram = mt_spectrogram.T

    # End timer and get elapsed compute time
    toc = timeit.default_timer()
    if verbose:
        print("\n Multitaper compute time: " + "%.2f" % (toc - tic) + " seconds")

    if np.all(mt_spectrogram.flatten() == 0):
        print("\n Data was all zeros, no output")

    # Plot multitaper spectrogram
    if plot_on:
        # convert from power to dB
        spect_data = nanpow2db(mt_spectrogram)

        # Set x and y axes
        dx = stimes[1] - stimes[0]
        dy = sfreqs[1] - sfreqs[0]
        extent = [stimes[0]-dx, stimes[-1]+dx, sfreqs[-1]+dy, sfreqs[0]-dy]

        # Plot spectrogram
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        im = ax.imshow(spect_data, extent=extent, aspect='auto')
        fig.colorbar(im, ax=ax, label='PSD (dB)', shrink=0.8)
        ax.set_xlabel("Time (HH:MM:SS)")
        ax.set_ylabel("Frequency (Hz)")
        im.set_cmap(plt.cm.get_cmap('cet_rainbow4'))
        ax.invert_yaxis()

        # Scale colormap
        if clim_scale:
            clim = np.percentile(spect_data, [5, 98])  # from 5th percentile to 98th
            im.set_clim(clim)  # actually change colorbar scale

        fig.show()
        if return_fig:
            return mt_spectrogram, stimes, sfreqs, (fig, ax)

    return mt_spectrogram, stimes, sfreqs


# Helper Functions #

# Process User Inputs #
def process_input(data, fs, frequency_range=None, time_bandwidth=5, num_tapers=None, window_params=None, min_nfft=0,
                  detrend_opt='linear', plot_on=True, verbose=True):
    """ Helper function to process multitaper_spectrogram() arguments
            Arguments:
                    data (1d np.array): time series data-- required
                    fs (float): sampling frequency in Hz  -- required
                    frequency_range (list): 1x2 list - [<min frequency>, <max frequency>] (default: [0 nyquist])
                    time_bandwidth (float): time-half bandwidth product (window duration*half bandwidth of main lobe)
                                            (default: 5 Hz*s)
                    num_tapers (int): number of DPSS tapers to use (default: None [will be computed
                                      as floor(2*time_bandwidth - 1)])
                    window_params (list): 1x2 list - [window size (seconds), step size (seconds)] (default: [5 1])
                    min_nfft (int): minimum allowable NFFT size, adds zero padding for interpolation (closest 2^x)
                                    (default: 0)
                    detrend_opt (string): detrend data window ('linear' (default), 'constant', 'off')
                                          (Default: 'linear')
                    plot_on (True): plot results (default: True)
                    verbose (True): display spectrogram properties (default: true)
            Returns:
                    data (1d np.array): same as input
                    fs (float): same as input
                    frequency_range (list): same as input or calculated from fs if not given
                    time_bandwidth (float): same as input or default if not given
                    num_tapers (int): same as input or calculated from time_bandwidth if not given
                    winsize_samples (int): number of samples in single time window
                    winstep_samples (int): number of samples in a single window step
                    window_start (1xm np.array): array of timestamps representing the beginning time for each window
                    num_windows (int): number of windows in the data
                    nfft (int): length of signal to calculate fft on
                    detrend_opt ('string'): same as input or default if not given
                    plot_on (bool): same as input
                    verbose (bool): same as input
    """

    # Make sure data is 1 dimensional np array
    if len(data.shape) != 1:
        if (len(data.shape) == 2) & (data.shape[1] == 1):  # if it's 2d, but can be transferred to 1d, do so
            data = np.ravel(data[:, 0])
        elif (len(data.shape) == 2) & (data.shape[0] == 1):  # if it's 2d, but can be transferred to 1d, do so
            data = np.ravel(data.T[:, 0])
        else:
            raise TypeError("Input data is the incorrect dimensions. Should be a 1d array with shape (n,) where n is \
                            the number of data points. Instead data shape was " + str(data.shape))

    # Set frequency range if not provided
    if frequency_range is None:
        frequency_range = [0, fs / 2]

    # Set detrending method
    detrend_opt = detrend_opt.lower()
    if detrend_opt != 'linear':
        if detrend_opt in ['const', 'constant']:
            detrend_opt = 'constant'
        elif detrend_opt in ['none', 'false', 'off']:
            detrend_opt = 'off'
        else:
            raise ValueError("'" + str(detrend_opt) + "' is not a valid argument for detrend_opt. The choices " +
                             "are: 'constant', 'linear', or 'off'.")
    # Check if frequency range is valid
    if frequency_range[1] > fs / 2:
        frequency_range[1] = fs / 2
        warnings.warn('Upper frequency range greater than Nyquist, setting range to [' +
                      str(frequency_range[0]) + ', ' + str(frequency_range[1]) + ']')

    # Set number of tapers if none provided
    if num_tapers is None:
        num_tapers = math.floor(2 * time_bandwidth) - 1

    # Warn if number of tapers is suboptimal
    if num_tapers != math.floor(2 * time_bandwidth) - 1:
        warnings.warn('Number of tapers is optimal at floor(2*TW) - 1. consider using ' +
                      str(math.floor(2 * time_bandwidth) - 1))

    # If no window params provided, set to defaults
    if window_params is None:
        window_params = [5, 1]

    # Check if window size is valid, fix if not
    if window_params[0] * fs % 1 != 0:
        winsize_samples = round(window_params[0] * fs)
        warnings.warn('Window size is not divisible by sampling frequency. Adjusting window size to ' +
                      str(winsize_samples / fs) + ' seconds')
    else:
        winsize_samples = window_params[0] * fs

    # Check if window step is valid, fix if not
    if window_params[1] * fs % 1 != 0:
        winstep_samples = round(window_params[1] * fs)
        warnings.warn('Window step size is not divisible by sampling frequency. Adjusting window step size to ' +
                      str(winstep_samples / fs) + ' seconds')
    else:
        winstep_samples = window_params[1] * fs

    # Get total data length
    len_data = len(data)

    # Check if length of data is smaller than window (bad)
    if len_data < winsize_samples:
        raise ValueError("\nData length (" + str(len_data) + ") is shorter than window size (" +
                         str(winsize_samples) + "). Either increase data length or decrease window size.")

    # Find window start indices and num of windows
    window_start = np.arange(0, len_data - winsize_samples + 1, winstep_samples)
    num_windows = len(window_start)

    # Get num points in FFT
    if min_nfft == 0:  # avoid divide by zero error in np.log2(0)
        nfft = max(2 ** math.ceil(np.log2(abs(winsize_samples))), winsize_samples)
    else:
        nfft = max(max(2 ** math.ceil(np.log2(abs(winsize_samples))), winsize_samples),
                   2 ** math.ceil(np.log2(abs(min_nfft))))

    return ([data, fs, frequency_range, time_bandwidth, num_tapers,
             int(winsize_samples), int(winstep_samples), window_start, num_windows, nfft,
             detrend_opt, plot_on, verbose])


# PROCESS THE SPECTROGRAM PARAMETERS #
def process_spectrogram_params(fs, nfft, frequency_range, window_start, datawin_size):
    """ Helper function to create frequency vector and window indices
        Arguments:
             fs (float): sampling frequency in Hz  -- required
             nfft (int): length of signal to calculate fft on -- required
             frequency_range (list): 1x2 list - [<min frequency>, <max frequency>] -- required
             window_start (1xm np array): array of timestamps representing the beginning time for each
                                          window -- required
             datawin_size (float): seconds in one window -- required
        Returns:
            window_idxs (nxm np array): indices of timestamps for each window
                                        (nxm where n=number of windows and m=datawin_size)
            stimes (1xt np array): array of times for the center of the spectral bins
            sfreqs (1xf np array): array of frequency bins for the spectrogram
            freq_inds (1d np array): boolean array of which frequencies are being analyzed in
                                      an array of frequencies from 0 to fs with steps of fs/nfft
    """

    # create frequency vector
    df = fs / nfft
    sfreqs = np.arange(0, fs, df)

    # Get frequencies for given frequency range
    freq_inds = (sfreqs >= frequency_range[0]) & (sfreqs <= frequency_range[1])
    sfreqs = sfreqs[freq_inds]

    # Compute times in the middle of each spectrum
    window_middle_samples = window_start + round(datawin_size / 2)
    stimes = window_middle_samples / fs

    # Get indexes for each window
    window_idxs = np.atleast_2d(window_start).T + np.arange(0, datawin_size, 1)
    window_idxs = window_idxs.astype(int)

    return [window_idxs, stimes, sfreqs, freq_inds]


# DISPLAY SPECTROGRAM PROPERTIES
def display_spectrogram_props(fs, time_bandwidth, num_tapers, data_window_params, frequency_range, nfft, detrend_opt):
    """ Prints spectrogram properties
        Arguments:
            fs (float): sampling frequency in Hz  -- required
            time_bandwidth (float): time-half bandwidth product (window duration*1/2*frequency_resolution) -- required
            num_tapers (int): number of DPSS tapers to use -- required
            data_window_params (list): 1x2 list - [window length(s), window step size(s)] -- required
            frequency_range (list): 1x2 list - [<min frequency>, <max frequency>] -- required
            nfft(float): number of fast fourier transform samples -- required
            detrend_opt (str): detrend data window ('linear' (default), 'constant', 'off') -- required
        Returns:
            This function does not return anything
    """

    data_window_params = np.asarray(data_window_params) / fs

    # Print spectrogram properties
    print("Multitaper Spectrogram Properties: ")
    print('     Spectral Resolution: ' + str(2 * time_bandwidth / data_window_params[0]) + 'Hz')
    print('     Window Length: ' + str(data_window_params[0]) + 's')
    print('     Window Step: ' + str(data_window_params[1]) + 's')
    print('     Time Half-Bandwidth Product: ' + str(time_bandwidth))
    print('     Number of Tapers: ' + str(num_tapers))
    print('     Frequency Range: ' + str(frequency_range[0]) + "-" + str(frequency_range[1]) + 'Hz')
    print('     NFFT: ' + str(nfft))
    print('     Detrend: ' + detrend_opt + '\n')


# NANPOW2DB
def nanpow2db(y):
    """ Power to dB conversion, setting bad values to nans
        Arguments:
            y (float or array-like): power
        Returns:
            ydB (float or np array): inputs converted to dB with 0s and negatives resulting in nans
    """

    if isinstance(y, int) or isinstance(y, float):
        if y == 0:
            return np.nan
        else:
            ydB = 10 * np.log10(y)
    else:
        if isinstance(y, list):  # if list, turn into array
            y = np.asarray(y)
        y = y.astype(float)  # make sure it's a float array so we can put nans in it
        y[y == 0] = np.nan
        ydB = 10 * np.log10(y)

    return ydB


# Helper #
def is_outlier(data):
    smad = 1.4826 * np.median(abs(data - np.median(data)))  # scaled median absolute deviation
    outlier_mask = abs(data-np.median(data)) > 3*smad  # outliers are more than 3 smads away from median
    outlier_mask = (outlier_mask | np.isnan(data) | np.isinf(data))
    return outlier_mask


# CALCULATE MULTITAPER SPECTRUM ON SINGLE SEGMENT
def calc_mts_segment(data_segment, dpss_tapers, nfft, freq_inds, detrend_opt, num_tapers, dpss_eigen, weighting, wt):
    """ Helper function to calculate the multitaper spectrum of a single segment of data
        Arguments:
            data_segment (1d np.array): One window worth of time-series data -- required
            dpss_tapers (2d np.array): Parameters for the DPSS tapers to be used.
                                       Dimensions are (num_tapers, winsize_samples) -- required
            nfft (int): length of signal to calculate fft on -- required
            freq_inds (1d np array): boolean array of which frequencies are being analyzed in
                                      an array of frequencies from 0 to fs with steps of fs/nfft
            detrend_opt (str): detrend data window ('linear' (default), 'constant', 'off')
            num_tapers (int): number of tapers being used
            dpss_eigen (np array):
            weighting (str):
            wt (int or np array):
        Returns:
            mt_spectrum (1d np.array): spectral power for single window
    """

    # If segment has all zeros, return vector of zeros
    if all(data_segment == 0):
        ret = np.empty(sum(freq_inds))
        ret.fill(0)
        return ret

    if any(np.isnan(data_segment)):
        ret = np.empty(sum(freq_inds))
        ret.fill(np.nan)
        return ret

    # Option to detrend data to remove low frequency DC component
    if detrend_opt != 'off':
        data_segment = detrend(data_segment, type=detrend_opt)

    # Multiply data by dpss tapers (STEP 2)
    tapered_data = np.multiply(np.asmatrix(data_segment).T, np.asmatrix(dpss_tapers.T))

    # Compute the FFT (STEP 3)
    fft_data = np.fft.fft(tapered_data, nfft, axis=0)

    # Compute the weighted mean spectral power across tapers (STEP 4)
    spower = np.power(np.imag(fft_data), 2) + np.power(np.real(fft_data), 2)
    if weighting == 'adapt':
        # adaptive weights - for colored noise spectrum (Percival & Walden p368-370)
        tpower = np.dot(np.transpose(data_segment), (data_segment/len(data_segment)))
        spower_iter = np.mean(spower[:, 0:2], 1)
        spower_iter = spower_iter[:, np.newaxis]
        a = (1 - dpss_eigen) * tpower
        for i in range(3):  # 3 iterations only
            # Calc the MSE weights
            b = np.dot(spower_iter, np.ones((1, num_tapers))) / ((np.dot(spower_iter, np.transpose(dpss_eigen))) +
                                                                 (np.ones((nfft, 1)) * np.transpose(a)))
            # Calc new spectral estimate
            wk = (b**2) * np.dot(np.ones((nfft, 1)), np.transpose(dpss_eigen))
            spower_iter = np.sum((np.transpose(wk) * np.transpose(spower)), 0) / np.sum(wk, 1)
            spower_iter = spower_iter[:, np.newaxis]

        mt_spectrum = np.squeeze(spower_iter)

    else:
        # eigenvalue or uniform weights
        mt_spectrum = np.dot(spower, wt)
        mt_spectrum = np.reshape(mt_spectrum, nfft)  # reshape to 1D

    return mt_spectrum[freq_inds]
