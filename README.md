# SSN-DNN-JAX
## Introduction

This report presents the implementation of a State-Space Model (SSM) and 1D Convolutional Neural Network (CNN) in JAX. The model combines Sequential Monte Carlo (SMC) sampling with deep learning for time-series analysis. The CNN is integrated to enhance particle resampling in the SMC algorithm, thereby improving state estimation accuracy.

JAX is a framework developed by Google that enables high-performance machine learning computations. It provides functionalities for automatic differentiation and just-in-time (JIT) compilation, making it ideal for numerical simulations such as SMC filtering and deep learning applications.

## Methodology

### 1. State-Space Model

The state-space model represents a dynamical system characterized by the equations:

**State transition equation:**

$$  x_{t+1} = A x_t + B u_t + ε $$

where:
- `A` (`jnp.ndarray`, shape `(dim_state, dim_state)`) represents the transition matrix.
- `B` (`jnp.ndarray`, shape `(dim_state,)`) is an external control term.
- `ε` is process noise sampled from a Gaussian distribution with variance `Q` (`jnp.ndarray`, shape `(1,)`).

**Observation equation:**

\[ y_t = C x_t + D + ν \]

where:
- `C` (`jnp.ndarray`, shape `(dim_obs, dim_state)`) is the observation matrix.
- `D` (`jnp.ndarray`, shape `(dim_obs,)`) is an offset.
- `ν` is Gaussian observation noise with variance `R` (`jnp.ndarray`, shape `(1,)`).

#### Implementation
A `State` data class is used to efficiently represent system parameters, enabling flexible simulation of different dynamical behaviors.
```python
@dataclass
class State:
    A: jnp.ndarray           # (dim_state, dim_state)
    B: jnp.ndarray           # (dim_state,)
    C: jnp.ndarray           # (dim_obs, dim_state)
    D: jnp.ndarray           # (dim_obs,)
    Q: jnp.ndarray           # (1,)
    R: jnp.ndarray           # (1,)
    R0: jnp.ndarray          # (1,)
    dim_state: int           # e.g., 1
    initial_mean: jnp.ndarray  # (dim_state,)
```
**Example Usage:**
```python
state = State(A=jnp.eye(2), B=jnp.zeros(2), C=jnp.array([[1, 0]]), D=jnp.zeros(1), 
              Q=jnp.array([0.1]), R=jnp.array([0.1]), R0=jnp.array([1.0]), dim_state=2, 
              initial_mean=jnp.zeros(2))
```

### 2. FFT-based Particle Analysis

Before processing particles through the CNN, Fast Fourier Transform (FFT) is applied to capture frequency-domain characteristics. This allows the network to learn periodic structures in the data, which is useful for time-series classification.

#### Implementation
The FFT is computed using `jax.numpy.fft.fft`, and the resulting magnitude spectrum is normalized to reduce dominant DC components and compress the dynamic range.
```python
def apply_fft_to_dparticle(dparticle: jnp.ndarray) -> jnp.ndarray:
    """
    Applies Fast Fourier Transform (FFT) to the input particle data, 
    normalizes the transformed values, and returns the processed result.
    
    Args:
        dparticle (jnp.ndarray): Input time-series particle data.

    Returns:
        jnp.ndarray: Normalized FFT-transformed data.
    """
    fft_complex = jnp.fft.fft(dparticle, axis=0)  # Compute FFT
    fft_magnitude = jnp.abs(fft_complex)          # Compute magnitude
    fft_log = jnp.log1p(fft_magnitude)            # Log transformation
    mean_val = jnp.mean(fft_log)                  # Compute mean
    std_val = jnp.std(fft_log) + 1e-6             # Compute std dev (avoid div by zero)
    fft_norm = (fft_log - mean_val) / std_val     # Normalize
    return fft_norm
```
**Example Usage:**
```python
dparticle = jnp.array([0.1, 0.5, 0.3, 0.7, 1.0])
fft_transformed = apply_fft_to_dparticle(dparticle)
```

### 3. Sequential Monte Carlo (SMC) Updates

SMC employs particle filtering for estimating latent states. Systematic resampling is implemented using Blackjax.
- If `y > α`, a Kalman filter-based update is applied.
- Otherwise, a standard Gaussian proposal is used.

#### Implementation
The `proposal_fn` function generates new particle estimates based on the current state and observations.
```python
def proposal_fn(state: State, particle: jnp.ndarray, y: float, key) -> (jnp.ndarray, float):
    """
    Generates a new particle proposal based on the given state and observation.

    Args:
        state (State): The current system state.
        particle (jnp.ndarray): The particle to update.
        y (float): The observed value.
        key: PRNG key for random number generation.

    Returns:
        tuple: A new particle and its log weight.
    """
    moved = jnp.dot(state.A, particle) + state.B
    threshold = 6.0

    def above_threshold(_):
        return moved, state.Q

    def below_threshold(_):
        return moved, state.R * 1.1

    mean_q, sigma_q = jax.lax.cond(y > threshold, above_threshold, below_threshold, operand=None)
    new_particle = mean_q + jnp.sqrt(sigma_q) * jr.normal(key, shape=mean_q.shape)
    log_weight = -0.5 * (((new_particle - mean_q) ** 2) / sigma_q + jnp.log(2 * math.pi * sigma_q))

    return new_particle, log_weight
```
**Example Usage:**
```python
key = random.PRNGKey(0)
new_particle, log_weight = proposal_fn(state, jnp.array([0.5, 0.2]), y=0.8, key=key)
```

The `smc_update2` function performs the update step, adjusting particle weights.
```python

def smc_update2(state: State, particle: jnp.ndarray, y: float, key) -> (jnp.ndarray, jnp.ndarray):
    """
    Performs a single update step in the Sequential Monte Carlo (SMC) framework.

    Args:
        state (State): The current system state.
        particle (jnp.ndarray): The current particle representation.
        y (float): The observed measurement.
        key: PRNG key for random number generation.

    Returns:
        tuple: Updated particle and corresponding weight.
    """
    new_particle, weight = proposal_fn(state, particle, y, key)
    return new_particle, weight
```
**Example Usage:**
```python
new_particle, weight = smc_update2(state, jnp.array([0.5, 0.2]), y=0.8, key=key)
```

The `particle_filter_single_trial2` function runs a single trial particle filter.
```python
def particle_filter_single_trial2(key, state: State, observations: jnp.ndarray, num_particles: int) -> jnp.ndarray:
    """
    Runs a single trial of the particle filter for state estimation.

    Args:
        key: PRNG key for random number generation.
        state (State): The system state parameters.
        observations (jnp.ndarray): The observed data sequence (T,).
        num_particles (int): The number of particles used in the filter.

    Returns:
        jnp.ndarray: History of particle states across time steps.
    """
    T = observations.shape[0]

    # Initialize particles based on initial state mean and noise variance
    init_particles = state.initial_mean + jnp.sqrt(state.R0) * jr.normal(
        key, shape=(num_particles, state.dim_state)
    )

    def step_fn(carry, y):
        key, particles = carry
        key, subkey = jr.split(key)

        # Perform SMC update on each particle
        new_particles, weights = jax.vmap(lambda p, k: smc_update2(state, p, y, k))(
            particles, jr.split(subkey, particles.shape[0])
        )

        # Normalize weights
        norm_weights = weights / (jnp.sum(weights) + 1e-6)

        # Systematic resampling
        indices = systematic(subkey, norm_weights, num_particles)
        resampled_particles = new_particles[indices]

        return (key, resampled_particles), resampled_particles

    # Run the particle filter over time using JAX scan
    _, particles_history = jax.lax.scan(step_fn, (key, init_particles), observations)

    return particles_history
```
**Example Usage:**
```python
particles_history = particle_filter_single_trial2(key, state, jnp.array([1.0, 2.0, 3.0]), num_particles=100)
```

### 4. CNN-based Particle Resampling

A 1D CNN is implemented in JAX using the Flax framework. The CNN architecture includes:
- Two convolutional layers (kernel sizes: 10, 20) with batch normalization and Leaky ReLU activation.
- Pooling layers for dominant feature extraction.
- Fully connected layers for classification (binary: class 0 or 1).
- Dropout layers to prevent overfitting.

#### Implementation
The `CNN1D` model defines a 1D CNN for time-series classification.
```python
class CNN1D(nn.Module):
    """
    A 1D Convolutional Neural Network (CNN) for time-series classification.

    Args:
        input_channels (int): Number of input channels.
        input_time_steps (int): Number of time steps in the input sequence.
        dropout_rate (float): Dropout rate for regularization (default: 0.3).

    Methods:
        __call__(x, train=True): Forward pass through the CNN.

    Returns:
        jnp.ndarray: Output logits with shape (batch, 2).
    """
    input_channels: int
    input_time_steps: int
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Forward pass of the 1D CNN.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch, time_steps, channels).
            train (bool): If True, applies dropout and batch normalization in training mode.

        Returns:
            jnp.ndarray: Output logits of shape (batch, 2).
        """
        # First convolutional block
        x = nn.Conv(features=8, kernel_size=(10,), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.leaky_relu(x)
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        # Second convolutional block
        x = nn.Conv(features=16, kernel_size=(20,), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.leaky_relu(x)
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        # Fully connected layers
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=64)(x)
        x = nn.leaky_relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        # Output layer
        x = nn.Dense(features=2)(x)

        return x
```
**Example Usage:**
```python
cnn_model = CNN1D(input_channels=1, input_time_steps=100, dropout_rate=0.1)
```

The `cnn_apply_fn` function applies the CNN model.
```python
def cnn_apply_fn(variables, inputs, train, **kwargs):
    """
    Applies the CNN model to the given inputs.

    Args:
        variables: Model parameters and state.
        inputs (jnp.ndarray): Input tensor of shape (batch, time_steps, channels).
        train (bool): Whether the model is in training mode.
        **kwargs: Additional arguments for the model.

    Returns:
        jnp.ndarray: Output logits of shape (batch, 2).
    """
    return cnn_model.apply(variables, inputs, train=train, **kwargs)
```
**Example Usage:**
```python
logits = cnn_apply_fn(cnn_variables, jnp.ones((10, 100, 1)), train=True)
```

### 5. Expectation-Maximization (EM) Algorithm

The EM algorithm iteratively refines the state-space model parameters:
- **E-Step:** Running the particle filter, updating particle weights based on CNN classification.
- **M-Step:** Updating the process noise variance (`Q`) using observed particle variance.

#### Implementation
The `run_em` function runs the EM algorithm with CNN updates.
```python
def run_em(key, state: State, observations_all_trials: jnp.ndarray, num_particles: int,
           num_iters: int, cnn_variables, cnn_opt_state, cnn_apply_fn, optimizer, cnn_labels):
    """
    Runs the Expectation-Maximization (EM) algorithm for state-space model estimation.

    Steps:
      - E–Step: Particle filtering across trials.
      - Apply FFT on particles.
      - CNN-based resampling per trial using FFT-transformed particles.
      - M–Step: Update process noise variance (Q).
      - Update CNN parameters using the new (FFT-transformed) particles.

    Args:
        key: PRNG key for randomness.
        state (State): The current system state.
        observations_all_trials (jnp.ndarray): Observations for multiple trials.
        num_particles (int): Number of particles for SMC.
        num_iters (int): Number of EM iterations.
        cnn_variables: CNN model parameters.
        cnn_opt_state: CNN optimizer state.
        cnn_apply_fn: Function to apply CNN model.
        optimizer: CNN optimizer.
        cnn_labels: Labels for CNN classification.

    Returns:
        tuple: Updated state, log-likelihoods, final particle history, CNN parameters, and optimizer state.
    """
    log_likelihoods = []
    final_particles_history = None

    for i in range(num_iters):
        key, subkey = jr.split(key)

        # Run particle filtering across trials
        particles_trials = particle_filter_multi_trial2(subkey, state, observations_all_trials, num_particles)
        # particles_trials shape: (num_trials, T, num_particles, dim_state)

        # CNN-based resampling using FFT-transformed particles
        num_trials = particles_trials.shape[0]
        keys_for_cnn = jr.split(key, num_trials)

        def resample_trial(particles, label, key_):
            return cnn_resample_particles(particles, cnn_variables, cnn_apply_fn, label, key_)

        particles_resampled = jax.vmap(resample_trial)(particles_trials, cnn_labels, keys_for_cnn)
        final_particles_history = particles_resampled

        # M-Step: Update process noise variance (Q)
        state = m_step(state, particles_resampled, damping=0.1)

        # Compute prediction error
        x_t = particles_resampled[:, :-1, :, :]
        x_tp1 = particles_resampled[:, 1:, :, :]
        predicted = jnp.einsum('ij,btpj->btpi', state.A, x_t) + state.B
        error = x_tp1 - predicted
        squared_error = jnp.sum(error**2, axis=-1)

        # Compute likelihood
        score = 0.5 * jnp.mean(squared_error)
        likelihood = jnp.exp(-score)
        log_likelihoods.append(likelihood)

        print(f"Iteration {i+1}: Score = {score}, Likelihood = {likelihood}, Updated Q = {state.Q}")

        # Apply FFT on each particle trajectory along the time axis
        fft_particles = jax.vmap(
            lambda trial: jax.vmap(lambda particle: apply_fft_to_dparticle(particle))(trial)
        )(particles_trials)
        # fft_particles shape: (num_trials, T, num_particles, dim_state)

        # Update CNN parameters using FFT-transformed particles
        cnn_variables, cnn_opt_state = nn_step(
            fft_particles, cnn_labels, cnn_variables, cnn_opt_state, cnn_apply_fn, optimizer,
            batch_size=32, epochs=5
        )

    return state, jnp.array(log_likelihoods), final_particles_history, cnn_variables, cnn_opt_state
```
**Example Usage:**
```python
state, log_likelihoods, particles_history, cnn_variables, cnn_opt_state = run_em(
    key, state, jnp.ones((5, 100, 1)), 100, 10, cnn_variables, cnn_opt_state, 
    cnn_apply_fn, optimizer, cnn_labels)
```

### 6. Cross-Validation and Evaluation

To ensure robustness, the model is evaluated using:
- **Dataset splitting:** Training and testing subsets (maintaining class balance).
- **Evaluation Metrics:** Receiver Operating Characteristic (ROC) curves, AUC scores.
- **Cross-validation:** Multiple training runs.

#### Implementation
The `evaluate_model_roc_curve` function evaluates CNN performance using ROC curves.
```python
def evaluate_model_roc_curve(cnn_variables, cnn_apply_fn, test_data, test_labels):
    """
    Evaluates the CNN model on the test set by applying the same FFT transformation
    (with log and normalization) that was used during training. The CNN then produces logits,
    from which the predicted probabilities and ROC curve are computed.

    Args:
        cnn_variables: CNN model parameters.
        cnn_apply_fn: Function to apply the CNN model.
        test_data (jnp.ndarray): Test dataset of shape (n_test, T, 1).
        test_labels (np.ndarray): Ground-truth binary labels (0 or 1) of shape (n_test,).

    Returns:
        tuple: False positive rates (fpr), true positive rates (tpr), and AUC score.
    """
    def fft_transform(sample):
        """
        Applies FFT transformation (log and normalization) to a single test sample.

        Args:
            sample (jnp.ndarray): Input sample of shape (T, 1).

        Returns:
            jnp.ndarray: Transformed sample of shape (T, 1).
        """
        sample = sample.squeeze(-1)  # Remove singleton channel dimension -> shape (T,)
        fft_sample = apply_fft_to_dparticle(sample)  # Apply FFT transformation -> shape (T,)
        return fft_sample[:, None]  # Add back the channel dimension -> shape (T, 1)

    # Apply FFT transformation to all test samples.
    fft_test_data = jax.vmap(fft_transform)(test_data)

    # Compute CNN logits.
    logits = cnn_apply_fn(cnn_variables, fft_test_data, train=False, mutable=False)

    # Compute class probabilities using softmax.
    probs = jax.nn.softmax(logits, axis=-1)

    # Extract positive-class probabilities.
    pos_probs = np.array(probs[:, 1])

    # Compute ROC curve and AUC score.
    fpr, tpr, thresholds = roc_curve(test_labels, pos_probs)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc
```
**Example Usage:**
```python
fpr, tpr, roc_auc = evaluate_model_roc_curve(cnn_variables, cnn_apply_fn, 
                                             jnp.ones((50, 100, 1)), jnp.ones(50))
```

## Conclusion

This implementation integrates SMC-based state estimation with deep learning (CNNs) in JAX, enhancing particle filtering accuracy. The FFT transformation further improves CNN performance by leveraging frequency-domain insights.
