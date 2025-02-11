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

**Example Usage:**
```python
key = random.PRNGKey(0)
new_particle, log_weight = proposal_fn(state, jnp.array([0.5, 0.2]), y=0.8, key=key)
```

The `smc_update2` function performs the update step, adjusting particle weights.

**Example Usage:**
```python
new_particle, weight = smc_update2(state, jnp.array([0.5, 0.2]), y=0.8, key=key)
```

The `particle_filter_single_trial2` function runs a single trial particle filter.

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

**Example Usage:**
```python
cnn_model = CNN1D(input_channels=1, input_time_steps=100, dropout_rate=0.1)
```

The `cnn_apply_fn` function applies the CNN model.

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
