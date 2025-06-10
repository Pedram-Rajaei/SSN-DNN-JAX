import jax
import jax.numpy as jnp
import jax.random as jr
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class State:
    """
    Container for the parameters of a linear Gaussian state–space model.

    Attributes
    ----------
    A : jnp.ndarray
        State transition matrix, shape (dim_state, dim_state).
    B : jnp.ndarray
        Control/offset vector, shape (dim_state,).
    C : jnp.ndarray
        Observation matrix, shape (dim_obs, dim_state).
    D : jnp.ndarray
        Observation offset, shape (dim_obs,).
    Q : jnp.ndarray
        Observation noise variance (as a 1-element array).
    R : jnp.ndarray
        State noise variance (as a 1-element array).
    R0 : jnp.ndarray
        Initial state variance (as a 1-element array).
    dim_state : int
        Dimension of the hidden state.
    initial_mean : jnp.ndarray
        Mean of the initial state distribution, shape (dim_state,).
    """
    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray
    D: jnp.ndarray
    Q: jnp.ndarray
    R: jnp.ndarray
    R0: jnp.ndarray
    dim_state: int
    initial_mean: jnp.ndarray

def systematic(key: jax.random.PRNGKey,
               weights: jnp.ndarray,
               num_particles: int) -> jnp.ndarray:
    """
    Perform systematic resampling given a set of normalized weights.

    Parameters
    ----------
    key : PRNGKey
        Random key for generating the uniform offset.
    weights : jnp.ndarray
        Array of shape (num_particles,) containing non-negative weights that sum to 1.
    num_particles : int
        Number of particles to resample.

    Returns
    -------
    indices : jnp.ndarray
        Integer array of shape (num_particles,) giving the selected particle indices.
    """
    # Uniform positions in [0,1) spaced by 1/num_particles
    offsets = (jnp.arange(num_particles) + jr.uniform(key, ())) / num_particles
    cumsum = jnp.cumsum(weights)
    indices = jnp.searchsorted(cumsum, offsets)
    return indices

def proposal_fn(state: State,
                particle: jnp.ndarray,
                y: float,
                key: jax.random.PRNGKey) -> (jnp.ndarray, float):
    """
    Importance proposal for a single particle in the E-step of EM via SMC.

    Moves the particle forward via the state model, then conditionally
    adjusts the mean and variance of the proposal based on the observation.

    Parameters
    ----------
    state : State
        Current SSM parameters.
    particle : jnp.ndarray
        Current particle state, shape (dim_state,).
    y : float
        Current observation (scalar).
    key : PRNGKey
        Random key for sampling.

    Returns
    -------
    new_particle : jnp.ndarray
        Sampled new particle, shape (dim_state,).
    log_weight : float
        Logarithm of the importance weight for this proposal.
    """
    # Predict via dynamics
    moved = state.A @ particle + state.B
    threshold = 6.0

    def above_threshold(_):
        # Update proposal variance & mean using Kalman-like formulas
        R_s, Q_s = state.R[0], state.Q[0]
        C_s, D_s = state.C[0, 0], state.D[0]
        sigma_q = 1.0 / ((1.0 / R_s) + (C_s**2 / Q_s))
        a = y - (C_s * moved[0] + D_s)
        mean_q = moved[0] + sigma_q * (C_s / Q_s) * a
        return jnp.array([mean_q]), jnp.array([sigma_q])

    def below_threshold(_):
        # If observation below threshold, use dynamics-only proposal
        return moved, state.R * 1.1

    mean_q, sigma_q = jax.lax.cond(y > threshold,
                                   above_threshold,
                                   below_threshold,
                                   operand=None)

    # Sample new particle and compute densities
    new_particle = mean_q + jnp.sqrt(sigma_q) * jr.normal(key, mean_q.shape)
    log_q = -0.5 * (((new_particle - mean_q)**2) / sigma_q + jnp.log(2*math.pi*sigma_q))
    log_trans = -0.5 * (((new_particle - moved)**2) / state.R + jnp.log(2*math.pi*state.R))

    def obs_above(_):
        Q_s, C_s, D_s = state.Q[0], state.C[0, 0], state.D[0]
        pred = C_s * new_particle[0] + D_s
        return -0.5 * (((y - pred)**2) / Q_s + jnp.log(2*math.pi*Q_s))

    log_obs = jax.lax.cond(y > threshold, obs_above, lambda _: 0.0, operand=None)
    log_weight = log_trans + log_obs - log_q
    return new_particle, log_weight


def normal_pdf(x: float, mean: float, var: float) -> float:
    """
    Evaluate the probability density of a univariate Gaussian.

    Parameters
    ----------
    x : float
        Point at which to evaluate the density.
    mean : float
        Gaussian mean.
    var : float
        Gaussian variance (must be > 0).

    Returns
    -------
    float
        pdf value at x.
    """
    return (1.0 / jnp.sqrt(2*math.pi*var)) * jnp.exp(-0.5 * ((x - mean)**2) / var)


def smc_update(state: State,
                particle: jnp.ndarray,
                y: float,
                key: jax.random.PRNGKey) -> (jnp.ndarray, float):
    """
    One SMC update for a single particle using an alternative proposal.

    Similar to `proposal_fn`, but returns weights in non-log form.

    Parameters
    ----------
    state : State
    particle : jnp.ndarray
        Current state (dim_state,).
    y : float
        Observation at current time.
    key : PRNGKey
        RNG key for proposal noise.

    Returns
    -------
    new_particle : jnp.ndarray
        Proposed particle (dim_state,).
    weight : float
        Unnormalized importance weight.
    """
    alpha = 1.1
    sigma_q = 1.0 / ((1.0 / state.R[0]) + (state.C[0,0]**2 / state.Q[0]))
    moved = state.A @ particle + state.B

    def branch_above(_):
        a = y - (state.C[0,0]*moved[0] + state.D[0])
        mean_q = moved[0] + sigma_q*(state.C[0,0]/state.Q[0])*a
        return jnp.array([mean_q]), sigma_q

    def branch_below(_):
        return moved, state.R[0]*alpha

    mean_q, cov = jax.lax.cond(y > 6, branch_above, branch_below, operand=None)
    new_particle = mean_q + jnp.sqrt(cov) * jr.normal(key, mean_q.shape)

    prop_prob = normal_pdf(new_particle[0], mean_q[0], cov)
    trans_prob = normal_pdf(new_particle[0], moved[0], state.R[0])

    def obs_branch(_):
        pred = state.C[0,0]*new_particle[0] + state.D[0]
        return normal_pdf(y, pred, state.Q[0])

    obs_prob = jax.lax.cond(y > 6, obs_branch, lambda _: 1.0, operand=None)
    weight = (trans_prob * obs_prob) / (prop_prob + 1e-6)
    return new_particle, weight


def particle_filter_single_trial(key: jax.random.PRNGKey,
                                  state: State,
                                  observations: jnp.ndarray,
                                  num_particles: int) -> jnp.ndarray:
    """
    Run a sequential particle filter on a single trial.

    Parameters
    ----------
    key : PRNGKey
        RNG key for initialization and resampling.
    state : State
        Model parameters.
    observations : jnp.ndarray
        Sequence of T observations, shape (T,).
    num_particles : int
        Number of particles.

    Returns
    -------
    particles_history : jnp.ndarray
        All particles over time, shape (T, num_particles, dim_state).
    """
    T = observations.shape[0]
    init_particles = state.initial_mean + jnp.sqrt(state.R0) * jr.normal(key, (num_particles, state.dim_state))

    def step_fn(carry, y_t):
        key, particles = carry
        key, subkey = jr.split(key)
        # Propose & weight
        new_particles, weights = jax.vmap(lambda p, k: smc_update(state, p, y_t, k))(
            particles, jr.split(subkey, particles.shape[0])
        )
        # Normalize & resample
        norm_w = weights / (jnp.sum(weights) + 1e-6)
        idx = systematic(subkey, norm_w, num_particles)
        resampled = new_particles[idx]
        return (key, resampled), resampled

    (_, _), history = jax.lax.scan(step_fn, (key, init_particles), observations)
    return history  # shape (T, num_particles, dim_state)


def particle_filter_multi_trial(key: jax.random.PRNGKey,
                                 state: State,
                                 observations_all_trials: jnp.ndarray,
                                 num_particles: int) -> jnp.ndarray:
    """
    Vectorized particle filter over multiple independent trials.

    Parameters
    ----------
    key : PRNGKey
    state : State
    observations_all_trials : jnp.ndarray
        Array of shape (num_trials, T) with observations per trial.
    num_particles : int

    Returns
    -------
    particles_trials : jnp.ndarray
        Array of shape (num_trials, T, num_particles, dim_state).
    """
    def filter_one(trial_key, obs):
        return particle_filter_single_trial(trial_key, state, obs, num_particles)

    keys = jr.split(key, observations_all_trials.shape[0])
    return jax.vmap(filter_one)(keys, observations_all_trials)


def m_step(state: State,
           particles_trials: jnp.ndarray,
           damping: float = 0.1,
           eps: float = 1e-6) -> State:
    """
    M-step update for the process noise variance Q in the SSM.

    Computes the expected one-step-ahead prediction error variance
    over all particles and trials, then damps the update.

    Parameters
    ----------
    state : State
    particles_trials : jnp.ndarray
        Array (num_trials, T, num_particles, dim_state).
    damping : float
        Mixing weight for new estimate vs. old value.
    eps : float
        Small floor to ensure positivity.

    Returns
    -------
    new_state : State
        New State with updated Q.
    """
    # Gather previous and next states
    x_t = particles_trials[:, :-1, :, :]
    x_tp1 = particles_trials[:, 1:, :, :]
    pred = jnp.einsum('ij,btpj->btpi', state.A, x_t) + state.B
    err = x_tp1 - pred
    mse = jnp.mean(err**2)
    new_Q = jnp.maximum(mse, eps)
    Q_updated = damping * new_Q + (1 - damping) * state.Q[0]

    return State(
        A=state.A,
        B=state.B,
        C=state.C,
        D=state.D,
        Q=jnp.array([Q_updated]),
        R=state.R,
        R0=state.R0,
        dim_state=state.dim_state,
        initial_mean=state.initial_mean
    )


def run_em(key: jax.random.PRNGKey,
           state: State,
           observations_all_trials: jnp.ndarray,
           num_particles: int,
           num_iters: int,
           cnn_variables: dict,
           cnn_opt_state,
           cnn_apply_fn: callable,
           optimizer,
           cnn_labels: jnp.ndarray,
           T_fixed: int):
    """
    Run the hybrid EM algorithm, alternating SMC (E-step) with CNN-based resampling
    and M-step updates of Q.

    Parameters
    ----------
    key : PRNGKey
    state : State
    observations_all_trials : jnp.ndarray
        Observations for all trials, shape (num_trials, T).
    num_particles : int
    num_iters : int
        Number of EM iterations.
    cnn_variables : dict
        Initial CNN parameters and state.
    cnn_opt_state
        Initial optimizer state for the CNN.
    cnn_apply_fn : callable
        CNN apply function.
    optimizer
        Optax optimizer for CNN training.
    cnn_labels : jnp.ndarray
        True labels for each trial, shape (num_trials,).
    T_fixed : int
        Fixed trajectory length for CNN inputs.

    Returns
    -------
    state : State
        Final estimated SSM parameters.
    log_likelihoods : jnp.ndarray
        Array of shape (num_iters,) with approximate likelihoods.
    final_particles_history : jnp.ndarray
        Resampled particles after last E-step (num_trials, T, num_particles, dim_state).
    cnn_variables : dict
        Final CNN parameters and state.
    cnn_opt_state
        Final CNN optimizer state.
    error_list : list of float
        Training error per iteration.
    """
    log_likelihoods = []
    error_list = []
    final_particles_history = None

    # Dummy input for activation visualization
    dummy_input = compute_particle_spectrum(jnp.ones((T_fixed,)), T_fixed)

    for i in range(num_iters):
        key, subkey = jr.split(key)
        # E-step: particle filtering
        particles = particle_filter_multi_trial(subkey, state, observations_all_trials, num_particles)
        # CNN-based resampling
        trials = particles.shape[0]
        keys_cnn = jr.split(key, trials)
        particles = jax.vmap(lambda p, lbl, k: cnn_resample_particles(p, cnn_variables, cnn_apply_fn, lbl, k, T_fixed))(
            particles, cnn_labels, keys_cnn
        )
        final_particles_history = particles

        # M-step: update Q
        state = m_step(state, particles, damping=0.1)

        # Compute track error & likelihood
        x_t = particles[:, :-1, :, :]
        x_tp1 = particles[:, 1:, :, :]
        pred = jnp.einsum('ij,btpj->btpi', state.A, x_t) + state.B
        err = x_tp1 - pred
        score = 0.5 * jnp.mean(err**2)
        lik = jnp.exp(-score)
        log_likelihoods.append(lik)
        error_list.append(score)
        print(f"Iteration {i+1}: Score={score}, Likelihood={lik}, Q={state.Q}")

        # CNN training on new spectra
        data_np = np.squeeze(particles, axis=-1)
        train_spec, freqs = compute_spectra_per_particle(data_np)
        train_spec = np.expand_dims(train_spec, axis=-1)
        key, cnn_key = jr.split(key)
        cnn_variables, cnn_opt_state, key = nn_step(
            train_spec, cnn_labels, cnn_variables, cnn_opt_state,
            cnn_apply_fn, optimizer, cnn_key, batch_size=32, epochs=25
        )

        if i < 5 or i == num_iters - 1:
            print(f"\n--- Visualization Iter {i+1} ---")
            print_cnn_summary(cnn_variables)
            plot_conv_weights(cnn_variables)
            plot_intermediate_activations(cnn_variables, cnn_apply_fn, dummy_input)

    # Plot convergence of training error
    plt.figure(figsize=(8,4))
    plt.plot(range(1, num_iters+1), jax.device_get(jnp.array(error_list)), marker='o', linestyle='-')
    plt.xlabel("EM Iteration")
    plt.ylabel("Train Error (Score)")
    plt.title("Error Convergence")
    plt.grid(True)
    plt.show()

    return state, jnp.array(log_likelihoods), final_particles_history, cnn_variables, cnn_opt_state, error_list
