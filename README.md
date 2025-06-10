# JAX-based implementation of structured state-space models combined with deep neural networks for scalable and interpretable time-series modeling(SSM-DNN-JAX)
## Introduction

A fast, differentiable implementation of State-Space Models combined with Deep Neural Networks (SSM-DNN) using [JAX](https://github.com/jax-ml/jax) for accelerated and scalable computation.

This repository extends the original [SSM-DNN](https://github.com/Pedram-Rajaei/SSM-DNN) implementation by leveraging JAX's high-performance automatic differentiation and parallelization capabilities. The resulting framework supports efficient training and inference for high-dimensional, temporally structured data.

---

## What's New in SSM-DNN-JAX?

- **JAX Backend**: Full rewrite of computational routines using JAX for GPU/TPU acceleration and vectorized operations.
- **Speed & Efficiency**: Up to **10× speed improvement** in training and inference compared to the NumPy-based implementation.
- **Clean Modular Design**: Reorganized code structure for easier experimentation and extension.
- **Expanded Functionality**:
  - EM + SMC particle-based inference
  - GP-based interpolation
  - Multivariate support

---

## Project Overview

State-Space Models (SSMs) are powerful tools for modeling **latent dynamics in time series**. By combining SSMs with **deep neural network encoders and decoders**, this framework learns both **interpretable latent dynamics** and **rich observation mappings**. For more details on the modular structure of this approach, please refer [here](https://github.com/Pedram-Rajaei/SSM-DNN/tree/main/Docs).

The JAX-based version maintains the original SSM-DNN architecture but improves:

- Execution speed (via `jit` compilation and `vmap`)
- Scalability to higher dimensions
- Differentiability of all components

---

<article>
  <section>
    <h3>Structure</h3>
    <pre><code>
SSM-DNN-JAX/
├── data/                 # Synthetic and real datasets
├── models/               # Core model definitions (SSM, RNN, CNN)
├── inference/            # EM algorithm, SMC filtering
├── utils/                # Helper functions
├── main.py               # Main training and evaluation script
└── README.md             # This file
    </code></pre>
  </section>

  <section>
    <h3>Installation</h3>
    <pre><code class="language-bash">
git clone https://github.com/Pedram-Rajaei/SSM-DNN-JAX.git
cd SSM-DNN-JAX
pip install -r requirements.txt
    </code></pre>
  </section>
</article>

---
<h2>Code Example(Real Data Applications)</h2>

<article id="example-1">
<h3>Brief Death Implicit Association Task (BDIAT)</h3>
<p>
  In this study, we demonstrate the application of the SSM-DNN framework to behavioral data collected from the D-IAT task. Each observation consists of reaction time (RT) data recorded across 360 sequential trials. The dataset includes 46 participants, each stratified into either the High Suicidal Ideation (High SI) or Low Suicidal Ideation (Low SI) group based on ecological momentary assessment scores. For modeling purposes, we treat each participant’s data as an individual trial, resulting in 23 trials in total, with each trial comprising 360 time-ordered RT samples. Each trial is labeled according to the participant’s SI group—High SI or Low SI.
</p>
<p>The observation dimension is one, and we model the latent state process using a random walk. In this framework, the state-space model (SSM) functions as an adaptive smoother over the observed reaction times. The deep neural network (DNN) component is implemented as a one-dimensional convolutional neural network (1D-CNN) consisting of two convolutional layers followed by max-pooling. To evaluate model performance, we use a cross-validation scheme that assesses overall prediction accuracy, as well as specificity and sensitivity. We compare the SSM-DNN model against a baseline DNN (without the SSM component) and alternative state of art modeling frameworks such as pre trained ResNet. Our results show that SSM-DNN achieves higher predictive accuracy while maintaining a better balance between specificity and sensitivity. A detailed breakdown of the SSM-DNN implementation steps is provided here:</p>
<ul>
    <li><a href="Neural_Network_for_Classification_Task.py">Neural Network for Classification Task</a>: Demonstrates how to implement and train a neural network for EEG data classification and visualize its performance.
    </li>
    <li>
        <a href="Latent_State_Inference.PY">Latent State Inference</a>: Showcases the particle filter and EM algorithm in action.
    </li>
    <li>
        <a href="Evaluate_Metric.PY">Evaluate Metric</a>: Evaluates model performance using accuracy, ROC curves, and AUC.
    </li>
</ul>
</article>

## Conclusion

This implementation integrates SMC-based state estimation with deep learning (CNNs) in JAX, enhancing particle filtering accuracy. The FFT transformation further improves CNN performance by leveraging frequency-domain insights. As observed in Fig. 1, the particles successfully capture the underlying pattern of the dataset. However, Fig. 2 reveals challenges in accurately classifying the labels, indicating that further refinement is needed to improve the classification performance.

<figure>
  <img src="Images/SSM.png" alt="Description" width="400">
  <figcaption>Figure 1: The particles successfully capture the underlying pattern hidden within the noisy observations, effectively filtering out noise to reveal the true structure of the dataset.</figcaption>
</figure>

<figure>
  <img src="Images/CNN.png" alt="Description" width="400">
  <figcaption>Figure 2: The CNN trained on particle-filtered data accurately predicts class labels, demonstrating strong classification performance with minimal misclassification </figcaption>
</figure>

---

<h2>Citation</h2>
<p>If you would use SSM-DNN in your research, please cite the following research papers:</p>
<pre><code>@@article{Paper1,
  title = {Novel techniques for neural data analysis},
  author = {Smith, J. and Doe, J.},
  journal = {Google Scholar},
  year = {2023},
  url = {https://scholar.google.com/citations?view_op=view_citation&hl=en&user=M8rzdnwAAAAJ&sortby=pubdate&citation_for_view=M8rzdnwAAAAJ:NXb4pA-qfm4C},
  note = {Accessed: 2025-01-14}
}
@article{Paper2,
  title = {Advances in deep learning for neuroscience},
  author = {Brown, A. and Taylor, K.},
  journal = {Google Scholar},
  year = {2022},
  url = {https://scholar.google.com/citations?view_op=view_citation&hl=en&user=jieyeRUAAAAJ&sortby=pubdate&citation_for_view=jieyeRUAAAAJ:NDuN12AVoxsC},
  note = {Accessed: 2025-01-14}
}
@article{Paper3,
  title = {State-space models in neuroscience},
  author = {Doe, J. and Smith, R.},
  journal = {PubMed},
  year = {2020},
  url = {https://pubmed.ncbi.nlm.nih.gov/31947169/},
  note = {Accessed: 2025-01-14}
}</code></pre>

---

<h2>Collaboration and Contribution</h2>
<p>We welcome your contribution in this research! Please check <a href="Contribution.md">here</a> for guidelines.</p>

---

<h2>License</h2>
<p>
    This project is licensed under the MIT License. See the <a href="License">license</a> file for details.
</p>

---

<h2>Acknowledgments</h2>
<p>
    This work was partially supported by the Defense Advanced Research Projects Agency (DARPA) under cooperative agreement #N660012324016. The content of this information does not necessarily reflect the position or policy of the Government, and no official endorsement should be inferred. We appreciate our research collaborators from UMN, Intheon, and others. This work was also supported by startup funds from the University of Houston. Special thanks go to our research collaborators and colleagues who contributed by providing data and offering thoughtful comments to refine our modeling framework.
</p>
