import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def main():
    # -------------------------------------------------------------------------
    # 1. Load or prepare your input data and labels
    # -------------------------------------------------------------------------
    # Y_interpolated should be defined or loaded before running this script.
    # Similarly, label_SI, label_TF, label_mdd_ctr must be provided.
    # For example:
    #   Y_interpolated = np.load('Y_interpolated.npy')
    #   label_SI = np.load('labels_SI.npy')
    #   label_TF = np.load('labels_TF.npy')
    #   label_mdd_ctr = np.load('labels_mdd_ctr.npy')
    #
    # Here we assume they are already present in the global namespace:
    Y = np.asarray(Y_interpolated)  # interpolated trajectories: shape (n_trials, T)
    label_SI      # binary labels for SI task
    label_TF      # binary labels for TF task
    label_mdd_ctr # binary labels for MDD vs control

    # Fixed trajectory length for CNN input
    T_fixed = 360

    # Number of cross-validation folds
    num_cv = 5

    # We'll store per-fold results here
    all_pos_probs_list   = []   # positive-class probabilities on test sets
    all_test_labels_list = []   # true labels from test sets
    cumulative_cm        = None # confusion matrix summed over all folds

    # For final aggregated ROC
    os.makedirs("figures", exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Initialize CNN with a dummy trajectory to compute reduced time dimension
    # -------------------------------------------------------------------------
    # Create a dummy 1D trajectory of length T_fixed to compute its spectrum
    dummy_traj = jnp.ones((T_fixed,))
    dummy_spec = compute_particle_spectrum(dummy_traj, T_fixed)
    # reduced_time = number of frequency bins after masking
    reduced_time = dummy_spec.shape[1]

    # Instantiate the 1D CNN model and optimizer
    cnn_model      = CNN1D(input_channels=1, input_time_steps=reduced_time)
    cnn_variables  = cnn_model.init(jr.PRNGKey(0),      # PRNG key for initialization
                                    dummy_spec,         # dummy input
                                    train=True)
    optimizer      = optax.adam(learning_rate=0.001, b1=0.9, b2=0.999)
    cnn_opt_state  = optimizer.init(cnn_variables)

    # (Optional) Visualize initial CNN weights and activations for debugging
    # print_cnn_summary(cnn_variables)
    # plot_conv_weights(cnn_variables)
    # plot_intermediate_activations(cnn_variables, cnn_apply_fn, dummy_spec)

    # -------------------------------------------------------------------------
    # 3. Cross-validation loop
    # -------------------------------------------------------------------------
    for cv in range(num_cv):
        print(f"\n=== CV iteration {cv+1}/{num_cv} ===")

        # Split data into training and test sets (test_size=4 trials per fold)
        y_train, label_si_train, y_test, label_si_test, train_idx, test_idx = \
            split_data_em(Y, label_SI, test_size=4)

        # Convert to JAX arrays for EM/CNN training
        observations_train = jnp.squeeze(jnp.asarray(y_train), axis=-1)  # shape: (n_train, T)
        cnn_labels_train   = jnp.asarray(label_si_train)                # SI labels for train
        observations_test  = jnp.asarray(y_test)                        # shape: (n_test, T)
        cnn_labels_test    = np.asarray(label_si_test)                  # SI labels for test

        # Also gather the other labels (TF and MDD/CTL) if needed for multi-task analysis
        label_tf_train      = label_TF[train_idx][:len(y_train)]
        label_mdd_ctr_train = label_mdd_ctr[train_idx][:len(y_train)]
        label_tf_test       = label_TF[test_idx]
        label_mdd_ctr_test  = label_mdd_ctr[test_idx]

        # Print shapes for verification
        print("Train observations:", observations_train.shape)
        print("Train SI labels   :", cnn_labels_train.shape)
        print("Train TF labels   :", label_tf_train.shape)
        print("Train MDD/CTL labels:", label_mdd_ctr_train.shape)
        print("Test observations :", observations_test.shape)
        print("Test SI labels    :", cnn_labels_test.shape)
        print("Test TF labels    :", label_tf_test.shape)
        print("Test MDD/CTL labels :", label_mdd_ctr_test.shape)

        # ---------------------------------------------------------------------
        # 3a. Initialize the State for the EM algorithm (linear-Gaussian model)
        # ---------------------------------------------------------------------
        dim_state    = 1
        dim_obs      = 1
        A = jnp.array([[1.0]])
        B = jnp.array([0.0])
        C = jnp.array([[1.0]])
        D = jnp.array([0.0])
        Q = jnp.array([0.5])     # process noise
        R = jnp.array([0.0005])  # observation noise
        R0 = jnp.array([1.0])    # initial noise
        initial_mean = jnp.array([7.0])
        state = State(
            A=A, B=B, C=C, D=D,
            Q=Q, R=R, R0=R0,
            dim_state=dim_state,
            initial_mean=initial_mean
        )

        # ---------------------------------------------------------------------
        # 3b. Prepare CNN input: compute per-particle spectra for training data
        # ---------------------------------------------------------------------
        # Create dummy particles for each training trial: shape (n_train, T, num_particles)
        dummy_particles = np.random.rand(observations_train.shape[0],
                                         observations_train.shape[1],
                                         200)
        # Compute spectra: output shape (n_train, freq_bins, num_particles)
        train_spectra, freqs = compute_spectra_per_particle(dummy_particles)
        # Keep only the low-frequency band of interest
        freq_mask = (freqs >= 0) & (freqs <= 0.035)
        train_spectra_reduced = train_spectra[:, freq_mask, :]
        # Add channel dimension for CNN: (n_train, reduced_time, 1)
        train_spectra_reduced = np.expand_dims(train_spectra_reduced, axis=-1)

        # Re-initialize the CNN for this fold (to reset weights)
        reduced_time = train_spectra_reduced.shape[1]
        cnn_model      = CNN1D(input_channels=1, input_time_steps=reduced_time)
        cnn_variables  = cnn_model.init(jr.PRNGKey(0), jnp.ones((1, reduced_time, 1)), train=True)
        optimizer      = optax.adam(learning_rate=0.001, b1=0.9, b2=0.999)
        cnn_opt_state  = optimizer.init(cnn_variables)

        # ---------------------------------------------------------------------
        # 3c. Run the EM algorithm with CNN updates for num_iters iterations
        # ---------------------------------------------------------------------
        em_key     = jr.PRNGKey(100 + cv)  # different seed each fold
        num_iters  = 60
        (updated_state,
         log_likes,
         final_particles_history,
         cnn_variables,
         cnn_opt_state,
         error_list) = run_em(
            em_key,
            state,
            observations_train,
            num_particles=200,
            num_iters=num_iters,
            cnn_variables=cnn_variables,
            cnn_opt_state=cnn_opt_state,
            cnn_apply_fn=cnn_apply_fn,
            optimizer=optimizer,
            cnn_labels=cnn_labels_train,
            T_fixed=T_fixed
        )
        print("Updated state Q after EM:", updated_state.Q)

        # ---------------------------------------------------------------------
        # 3d. Plot EM convergence (likelihood) and CNN training loss
        # ---------------------------------------------------------------------
        log_likes_np = jax.device_get(log_likes)
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, num_iters+1), log_likes_np, marker='o', linestyle='-')
        plt.xlabel("EM Iteration")
        plt.ylabel("Log-Likelihood")
        plt.title(f"Fold {cv+1}: EM Convergence")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"figures/likelihood_cv_{cv+1}.png")
        plt.close()

        error_list_np = jax.device_get(jnp.array(error_list))
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(error_list_np)+1), error_list_np, marker='o', linestyle='-')
        plt.xlabel("EM Iteration")
        plt.ylabel("CNN Training Loss")
        plt.title(f"Fold {cv+1}: CNN Loss Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"figures/training_loss_cv_{cv+1}.png")
        plt.close()

        # ---------------------------------------------------------------------
        # 3e. Evaluate on training set: compute ROC and accuracy
        # ---------------------------------------------------------------------
        train_pos_probs, train_true_labels, train_pred_labels, train_cm = \
            evaluate_model_test_set(
                cnn_variables,
                cnn_apply_fn,
                observations_train,
                cnn_labels_train,
                updated_state,
                num_particles=200,
                rng_key=jr.PRNGKey(1)
            )
        fpr_train, tpr_train, _    = roc_curve(train_true_labels, train_pos_probs)
        roc_auc_train              = auc(fpr_train, tpr_train)
        train_accuracy             = np.mean(train_pred_labels == train_true_labels) * 100

        # Plot Train ROC
        plt.figure(figsize=(8, 4))
        plt.plot(fpr_train, tpr_train, label=f'Train AUC = {roc_auc_train:.2f}')
        plt.plot([0,1], [0,1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Fold {cv+1}: Train ROC")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"figures/train_roc_cv_{cv+1}.png")
        plt.close()

        print(f"Fold {cv+1} Train Accuracy: {train_accuracy:.2f}%")

        # ---------------------------------------------------------------------
        # 3f. Evaluate on test set: compute ROC and accumulate confusion matrix
        # ---------------------------------------------------------------------
        test_pos_probs, test_true_labels, test_pred_labels, cm = \
            evaluate_model_test_set(
                cnn_variables,
                cnn_apply_fn,
                observations_test,
                cnn_labels_test,
                updated_state,
                num_particles=200,
                rng_key=jr.PRNGKey(0)
            )
        fpr_test, tpr_test, _ = roc_curve(test_true_labels, test_pos_probs)
        roc_auc_test          = auc(fpr_test, tpr_test)

        # Plot Test ROC
        plt.figure(figsize=(8, 4))
        plt.plot(fpr_test, tpr_test, label=f'Test AUC = {roc_auc_test:.2f}')
        plt.plot([0,1], [0,1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Fold {cv+1}: Test ROC")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"figures/test_roc_cv_{cv+1}.png")
        plt.close()

        print(f"Fold {cv+1} Confusion Matrix:\n{cm}")

        # Accumulate per-fold results
        all_pos_probs_list.append(test_pos_probs)
        all_test_labels_list.append(test_true_labels)
        cumulative_cm = cm if cumulative_cm is None else (cumulative_cm + cm)

    # -------------------------------------------------------------------------
    # 4. Aggregate results across all folds and plot overall ROC
    # -------------------------------------------------------------------------
    all_pos_probs = np.concatenate(all_pos_probs_list, axis=0)
    all_test_labels = np.concatenate(all_test_labels_list, axis=0)

    print("Aggregated confusion matrix over all folds:")
    print(cumulative_cm)

    fpr, tpr, _ = roc_curve(all_test_labels, all_pos_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 4))
    plt.plot(fpr, tpr, label=f'Aggregated Test AUC = {roc_auc:.2f}')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Aggregated Test ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    agg_roc_path = "figures/aggregated_test_roc.png"
    plt.savefig(agg_roc_path)
    plt.close()

    print(f"Saved aggregated ROC curve to {agg_roc_path}")
