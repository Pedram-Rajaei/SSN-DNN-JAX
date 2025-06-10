import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Tuple

def evaluate_model_test_set(cnn_variables: dict,
                            cnn_apply_fn: callable,
                            test_data: jnp.ndarray,
                            test_labels: np.ndarray,
                            state: State,
                            num_particles: int,
                            eval_key: jax.random.PRNGKey
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the CNN–augmented particle filter on held-out test trials.

    This function:
      1. Runs the particle filter on each test trial to generate trajectories.
      2. Converts each particle’s trajectory to an FFT-based spectrogram.
      3. Feeds every particle’s spectrogram through the trained CNN.
      4. Averages the CNN’s softmax probabilities over particles per trial.
      5. Computes predicted labels and the confusion matrix.

    Parameters
    ----------
    cnn_variables : dict
        Trained Flax CNN variables (parameters + state).
    cnn_apply_fn : callable
        The `apply` function for the CNN, signature:
        `logits = cnn_apply_fn(variables, inputs, train=False, mutable=False)`.
    test_data : jnp.ndarray
        Array of shape (n_test, T) containing test trial observations.
    test_labels : np.ndarray
        Integer array of true labels for each test trial, shape (n_test,).
    state : State
        Current state-space model parameters for particle filtering.
    num_particles : int
        Number of particles to use per trial in the filter.
    eval_key : jax.random.PRNGKey
        RNG key for stochastic resampling in the particle filter.

    Returns
    -------
    pos_probs : np.ndarray
        Array of positive-class probabilities for each test trial, shape (n_test,).
    true_labels : np.ndarray
        The original `test_labels`, shape (n_test,).
    predicted_labels : np.ndarray
        Integer array of predicted labels (0 or 1) per trial, shape (n_test,).
    cm : np.ndarray
        Confusion matrix of shape (2, 2), comparing true vs. predicted labels.
    """
    # 1) Particle filter on all test trials
    particles_trials = particle_filter_multi_trial2(
        eval_key, state, test_data, num_particles
    )  # shape: (n_test, T, num_particles, dim_state)

    # 2) Remove state-dimension and compute spectrograms per particle
    data_multitaper = np.squeeze(particles_trials, axis=-1)
    #    -> shape: (n_test, T, num_particles)
    spectra, freqs = compute_spectra_per_particle(data_multitaper)
    #    -> spectra shape: (n_test, reduced_freq, num_particles)
    # Expand to CNN input format: add channel dim
    spectra_in = np.expand_dims(spectra, axis=-1)
    #    -> shape: (n_test, reduced_freq, num_particles, 1)

    # 3) Rearrange so that particles become separate examples
    #    From (n_test, freq, num_particles, 1)
    #    to   (n_test, num_particles, freq, 1)
    fft_particles = jnp.transpose(spectra_in, (0, 2, 1, 3))
    n_test = fft_particles.shape[0]
    p_per_trial = fft_particles.shape[1]

    # 4) Flatten batch & particle dims for CNN
    flat_inputs = fft_particles.reshape(
        n_test * p_per_trial,
        fft_particles.shape[2],
        fft_particles.shape[3]
    )  # shape: (n_test * num_particles, freq, 1)

    # 5) Forward through CNN and softmax
    logits = cnn_apply_fn(cnn_variables, flat_inputs, train=False, mutable=False)
    probs = jax.nn.softmax(logits, axis=-1)

    # 6) Reshape back to (n_test, num_particles, num_classes)
    probs_np = np.array(probs).reshape(n_test, p_per_trial, -1)

    # 7) Average over particles to get trial‐level probabilities
    avg_probs = probs_np.mean(axis=1)          # shape: (n_test, 2)
    pos_probs = avg_probs[:, 1]                # positive‐class probability

    # 8) Predicted labels & confusion matrix
    predicted_labels = np.argmax(avg_probs, axis=-1)
    cm = confusion_matrix(test_labels, predicted_labels)

    return pos_probs, test_labels, predicted_labels, cm
