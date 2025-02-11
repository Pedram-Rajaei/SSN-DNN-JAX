# SSN-DNN-JAX
## Introduction

This report presents the implementation of a State-Space Model (SSM) and 1D Convolutional Neural Network (CNN) in JAX. The model combines Sequential Monte Carlo (SMC) sampling with deep learning for time-series analysis. The CNN is integrated to enhance particle resampling in the SMC algorithm, thereby improving state estimation accuracy.

JAX is a framework developed by Google that enables high-performance machine learning computations. It provides functionalities for automatic differentiation and just-in-time (JIT) compilation, making it ideal for numerical simulations such as SMC filtering and deep learning applications.

## Methodology

### 1. State-Space Model

The state-space model represents a dynamical system characterized by the equations:

**State transition equation:**

\[ x_{t+1} = A x_t + B u_t + \epsilon \]

where:
- `A` (`jnp.ndarray`, shape `(dim_state, dim_state)`) represents the transition matrix.
- `B` (`jnp.ndarray`, shape `(dim_state,)`) is an external control term.
- `\epsilon` is process noise sampled from a Gaussian distribution with variance `Q` (`jnp.ndarray`, shape `(1,)`).

**Observation equation:**

\[ y_t = C x_t + D + \nu \]

where:
- `C` (`jnp.ndarray`, shape `(dim_obs, dim_state)`) is the observation matrix.
- `D` (`jnp.ndarray`, shape `(dim_obs,)`) is an offset.
- `\nu` is Gaussian observation noise with variance `R` (`jnp.ndarray`, shape `(1,)`).

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

```
**Example Usage:**
```python
cnn_model = CNN1D(input_channels=1, input_time_steps=100, dropout_rate=0.1)
```

The `cnn_apply_fn` function applies the CNN model.
```python

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

```
**Example Usage:**
```python
fpr, tpr, roc_auc = evaluate_model_roc_curve(cnn_variables, cnn_apply_fn, 
                                             jnp.ones((50, 100, 1)), jnp.ones(50))
```

## Conclusion

This implementation integrates SMC-based state estimation with deep learning (CNNs) in JAX, enhancing particle filtering accuracy. The FFT transformation further improves CNN performance by leveraging frequency-domain insights.

## Future Work

- Extending to multi-class classification.
- Incorporating hierarchical Bayesian filtering.
- Experimenting with alternative deep learning architectures like RNNs.
