import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax import linen as nn


class CNN1D(nn.Module):
    """
    1D convolutional neural network for binary classification.

    Attributes
    ----------
    input_channels : int
        Number of feature channels in the input time series.
    input_time_steps : int
        Length (number of time steps) of the input series.
    dropout_rate : float, optional
        Dropout probability applied after each block (default is 0.3).

    Methods
    -------
    __call__(x, train=True)
        Applies the network to input data `x`, returning raw logits for two classes.
    """
    input_channels: int
    input_time_steps: int
    dropout_rate: float = 0.3  # You may further tune this value

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Forward pass of the 1D CNN.

        Parameters
        ----------
        x : jnp.ndarray
            Input batch of shape (batch_size, time_steps, input_channels).
        train : bool, optional
            Whether we are in training mode (controls BatchNorm & Dropout), by default True.

        Returns
        -------
        jnp.ndarray
            Logits of shape (batch_size, 2) for binary classification.
        """
        # Convolution Block 1
        x = nn.Conv(features=8, kernel_size=(4,), padding='SAME')(x)
        self.sow('intermediates', 'conv1', x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        # Convolution Block 2
        x = nn.Conv(features=16, kernel_size=(4,), padding='SAME')(x)
        self.sow('intermediates', 'conv2', x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        # Convolution Block 3
        x = nn.Conv(features=32, kernel_size=(4,), padding='SAME')(x)
        self.sow('intermediates', 'conv3', x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        # Global Average Pooling over time dimension
        x = jnp.mean(x, axis=1)

        # Dense Block
        x = nn.Dense(features=16)(x)
        self.sow('intermediates', 'dense1', x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        x = nn.Dense(features=2)(x)
        self.sow('intermediates', 'dense2', x)
        return x


def cnn_apply_fn(variables: dict,
                 inputs: jnp.ndarray,
                 train: bool,
                 **kwargs) -> tuple:
    """
    Apply the CNN model to inputs given a set of variables (parameters + state).

    Parameters
    ----------
    variables : dict
        A dict containing model parameters and batch statistics, as returned by `cnn_model.init(...)`.
    inputs : jnp.ndarray
        Input batch, shape (batch_size, time_steps, input_channels).
    train : bool
        Flag indicating training mode (enables dropout & updates batch stats).
    **kwargs
        Additional keyword args forwarded to `apply`, e.g. rngs={'dropout': key}.

    Returns
    -------
    tuple
        - logits: jnp.ndarray of shape (batch_size, 2)
        - new_model_state (if mutable stats requested): updated batch statistics
    """
    return cnn_model.apply(variables, inputs, train=train, **kwargs)


def cnn_loss_fn(variables: dict,
                batch_inputs: jnp.ndarray,
                batch_labels: jnp.ndarray,
                cnn_apply_fn: callable,
                dropout_rng: jax.random.PRNGKey) -> tuple:
    """
    Compute softmax cross-entropy loss for a batch, returning new batch_stats.

    Parameters
    ----------
    variables : dict
        Model params and state.
    batch_inputs : jnp.ndarray
        Input batch of shape (batch_size, time_steps, input_channels).
    batch_labels : jnp.ndarray
        Integer array of labels, shape (batch_size,), values in {0,1}.
    cnn_apply_fn : callable
        Function to call the model (e.g., the fn defined above).
    dropout_rng : PRNGKey
        Random key for dropout.

    Returns
    -------
    loss : jnp.ndarray
        Scalar loss (mean cross-entropy over the batch).
    new_model_state : dict
        Updated model state containing new batch statistics.
    """
    # Forward pass with dropout, tracking new batch_stats
    logits, new_model_state = cnn_apply_fn(
        variables,
        batch_inputs,
        train=True,
        rngs={'dropout': dropout_rng},
        mutable=['batch_stats']
    )
    # One-hot targets
    one_hot = jax.nn.one_hot(batch_labels, num_classes=2)
    # Mean cross-entropy loss
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    return loss, new_model_state


def cnn_train_step(variables: dict,
                   opt_state: optax.OptState,
                   batch_inputs: jnp.ndarray,
                   batch_labels: jnp.ndarray,
                   cnn_apply_fn: callable,
                   optimizer: optax.GradientTransformation,
                   dropout_rng: jax.random.PRNGKey) -> tuple:
    """
    Perform one optimization step on the CNN: compute gradients, update params & state.

    Parameters
    ----------
    variables : dict
        Current model parameters and state.
    opt_state : optax.OptState
        Current optimizer state.
    batch_inputs : jnp.ndarray
        Batch of inputs.
    batch_labels : jnp.ndarray
        Batch of integer labels.
    cnn_apply_fn : callable
        The apply function for the model.
    optimizer : optax.GradientTransformation
        The optimizer used for updates.
    dropout_rng : PRNGKey
        Random key for dropout.

    Returns
    -------
    new_variables : dict
        Updated model parameters and state (with updated 'batch_stats').
    new_opt_state : optax.OptState
        Updated optimizer state.
    loss : jnp.ndarray
        Scalar training loss for this batch.
    """
    # Compute loss & grads
    (loss, new_model_state), grads = jax.value_and_grad(
        cnn_loss_fn, has_aux=True
    )(
        variables,
        batch_inputs,
        batch_labels,
        cnn_apply_fn,
        dropout_rng
    )
    # Apply optimizer update
    updates, new_opt_state = optimizer.update(grads, opt_state, params=variables)
    updated_params = optax.apply_updates(variables, updates)

    # Replace only the batch_stats in the variables dict
    new_variables = updated_params.copy()
    new_variables['batch_stats'] = new_model_state['batch_stats']
    return new_variables, new_opt_state, loss


# JIT-compile train step for speed; args 4 & 5 (apply_fn & optimizer) are static.
cnn_train_step = jax.jit(cnn_train_step, static_argnums=(4, 5))


def nn_step(all_particles: jnp.ndarray,
            labels_train: jnp.ndarray,
            cnn_variables: dict,
            cnn_opt_state: optax.OptState,
            cnn_apply_fn: callable,
            optimizer: optax.GradientTransformation,
            key: jax.random.PRNGKey,
            batch_size: int = 32,
            epochs: int = 5) -> tuple:
    """
    Update CNN parameters by treating each particle-trajectory as a separate training example.

    Parameters
    ----------
    all_particles : jnp.ndarray
        Array of shape (num_trials, T, num_particles, dim_state) containing trajectories.
    labels_train : jnp.ndarray
        Integer labels for each trial, shape (num_trials,).
    cnn_variables : dict
        Current CNN parameters and state.
    cnn_opt_state : optax.OptState
        Current optimizer state.
    cnn_apply_fn : callable
        CNN apply function.
    optimizer : optax.GradientTransformation
        Optimizer to use.
    key : PRNGKey
        Random key for shuffling and dropout.
    batch_size : int, optional
        Number of trajectories per batch, by default 32.
    epochs : int, optional
        Number of passes through the dataset, by default 5.

    Returns
    -------
    cnn_variables : dict
        Updated model parameters and state.
    cnn_opt_state : optax.OptState
        Updated optimizer state.
    key : PRNGKey
        Updated random key.
    """
    num_trials, T, num_particles, dim_state = all_particles.shape

    # Reshape to (num_trials * num_particles, T, dim_state)
    x_temp = jnp.transpose(all_particles, (0, 2, 1, 3))
    x_temp = x_temp.reshape(-1, T, dim_state)

    # Repeat labels to match particles
    labels_repeated = jnp.repeat(labels_train, num_particles, axis=0)

    # Shuffle dataset
    dataset_size = x_temp.shape[0]
    key, perm_key = jr.split(key)
    perm = jax.random.permutation(perm_key, dataset_size)
    x_temp = x_temp[perm]
    labels_repeated = labels_repeated[perm]

    # Training loop
    rng = jr.PRNGKey(1234)
    num_batches = dataset_size // batch_size
    for _ in range(epochs):
        for i in range(num_batches):
            rng, dropout_rng = jr.split(rng)
            batch_inputs = x_temp[i*batch_size:(i+1)*batch_size]
            batch_labels = labels_repeated[i*batch_size:(i+1)*batch_size]
            cnn_variables, cnn_opt_state, loss = cnn_train_step(
                cnn_variables,
                cnn_opt_state,
                batch_inputs,
                batch_labels,
                cnn_apply_fn,
                optimizer,
                dropout_rng
            )
    return cnn_variables, cnn_opt_state, key


def cnn_resample_particles(all_particles: jnp.ndarray,
                           cnn_variables: dict,
                           cnn_apply_fn: callable,
                           trial_label: int,
                           key: jax.random.PRNGKey,
                           T_fixed: int) -> jnp.ndarray:
    """
    Resample a single trial’s particles according to CNN-predicted weights.

    For each particle:
      - Compute its spectrum using a fixed-length FFT.
      - Feed through the CNN (inference mode) to get a probability for the true class.
      - Normalize to obtain sampling weights.
      - Resample indices via systematic resampling.
      - Add small Gaussian jitter to avoid degeneracy.

    Parameters
    ----------
    all_particles : jnp.ndarray
        Array of shape (T, num_particles, dim_state) for one trial.
    cnn_variables : dict
        Trained CNN parameters and state.
    cnn_apply_fn : callable
        CNN apply function.
    trial_label : int
        True label of this trial (0 or 1).
    key : PRNGKey
        PRNGKey for resampling & jitter.
    T_fixed : int
        Fixed time length used when computing the spectrum.

    Returns
    -------
    jnp.ndarray
        Resampled particles of same shape (T, num_particles, dim_state),
        with small jitter added.
    """
    T, num_particles, dim_state = all_particles.shape

    def compute_particle_weight(i):
        # Extract 1D trajectory (shape T,)
        traj = all_particles[:, i, :].squeeze(-1)
        # Compute fixed-length spectrogram
        particle_spec = compute_particle_spectrum(traj, T_fixed)  # (1, freq_bins, 1)
        # Inference through CNN
        logits = cnn_apply_fn(cnn_variables, particle_spec, train=False)
        probs = jax.nn.softmax(logits, axis=-1)
        # Probability assigned to the true class
        return probs[0, trial_label]

    # Vectorize weight computation over particles
    weights_nn = jax.vmap(compute_particle_weight)(jnp.arange(num_particles))
    weight_sum = jnp.sum(weights_nn)

    # Normalize weights, handling NaNs or all-zero case
    weights_nn = jax.lax.cond(
        (jnp.isnan(weight_sum) | (weight_sum == 0)),
        lambda _: jnp.ones_like(weights_nn) / num_particles,
        lambda _: weights_nn / weight_sum,
        operand=None
    )

    # Systematic resampling
    resample_key, jitter_key = jr.split(key)
    indices = systematic(resample_key, weights_nn, num_particles)
    resampled = all_particles[:, indices, :]

    # Add tiny Gaussian jitter
    jitter = jr.normal(jitter_key, shape=resampled.shape) * 1e-5
    return resampled + jitter
