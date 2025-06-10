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
