# Robustness-of-Interpretability-Methods
## Origin
This implementation is based on the paper [On the Robustness of Interpretability Methods](https://arxiv.org/abs/1806.08049). It aim to quantify the robustness of an explanation model like e.g. LIME, using the definition of “local Lipschitz continuity”.

## What is the Lipschitz metric?
In essence, the Lipschitz metric describes the biggest magnification in the output when making small adjustments to the input of a function. This "function" will in our case be a method that accepts the input of a machine learning model and returns the feature contributions from an explainer model like LIME. By perturbing the input around a point while looking at how the output changes, we can calculate the Lipschitz metric (or L-metric), which is a non-negative real number. A small L-metric means that small changes in the input gives small changes in the output. This is often desirable for explainer models, where we normally would expect similar inputs to give somewhat similar explanations.

## How do I use it?
Functions for calculating the Lipschitz metric are found in `lipschitz_metric.py` along with some helper functions. For a demo of how to use this, look in the notebook `demo.ipynb`.
