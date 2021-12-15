import numpy as np
from typing import List, Tuple, Callable, Literal
from numpy.typing import ArrayLike

def lipschitz_metric(
    x0: ArrayLike,
    y0: ArrayLike,
    x_perturbations: ArrayLike,
    y_perturbations: ArrayLike,
    x_range: List[Tuple[float, float]] = None,
    y_range: List[Tuple[float, float]] = None,
    ) -> float:
    '''
    Metric describing the biggest change in output relative to change the change in input.
    The metric is the constant L in the definition of “local Lipschitz continuity”.
    The metric often gives more sense if the input variables are scaled to the same size. Use
    `x_range` to achieve this.
    Given a function y=f(x):
    - N is the number of inputs in `x`
    - M is the number of outputs in `y`
    - P is the number of perturbations

    :x0: Input vector with shape (1, N)
    :y0: Output vector with shape (1, M)
    :x_perturbations: Perturbed inputs in a matrix of shape (P,N)
    :y_perturbations: Output from each input, given as a matrix of shape (P,M)
    :x_range: Scaling of input. List of tuples specifying the min and max values of each individual input respectivly (e.g. [(0,1), (-10,10), ...]). Lenght equals N.
    :y_range: Scaling of outputs. List of tuples specifying the min and max values of each individual output respectivly (e.g. [(0,1), (-10,10), ...]). Length equals M.
    
    :return: (L, L_input). L is the Lipschitz metric, while L_input contains the x_perturbation that caused the L metric
    '''
    x0 = x0.copy()
    y0 = y0.copy()

    # Ensure correct shapes
    if x0.ndim == 1: x0 = x0.reshape(1,-1)
    if y0.ndim == 1: y0 = y0.reshape(1,-1)

    # Assert correct dimensions
    assert x0.shape[0] == y0.shape[0] == 1, f"x0 and y0 should have the shape (1,N) and (1,M) respectively, not (1,{x0.shape[0]}) and (1,{y0.shape[0]})"
    assert x0.shape[1] == x_perturbations.shape[1], f"x0 and x_permutations should both have N number of inputs, not {x0.shape[1]} and {x_perturbations.shape[1]}"
    assert y0.shape[1] == y_perturbations.shape[1], f"y0 and y_permutations should both have M number of inputs, not {y0.shape[1]} and {y_perturbations.shape[1]}"
    assert x_perturbations.shape[0] == y_perturbations.shape[0], f"x_perturbations and y_perturbations should containt the same number of permutations, not {x_perturbations.shape[0]} and {y_perturbations.shape[0]}"

    # Normalize variables
    if x_range or y_range:
        scale_transform = lambda x, x_min, x_max: (x - x_min) / (x_max - x_min)
        scale_inverse = lambda x, x_min, x_max: (x_max - x_min) * x + x_min

    if x_range:
        for i, (x_min, x_max) in enumerate(x_range):
            x0[:,i] = scale_transform(x0[:,i], x_min, x_max)
            x_perturbations[:,i] = scale_transform(x_perturbations[:,i], x_min, x_max)

    if y_range:
        for i, (y_min, y_max) in enumerate(y_range):
            y0[:,i] = scale_transform(y0[:,i], y_min, y_max)
            y_perturbations[:,i] = scale_transform(y_perturbations[:,i], y_min, y_max)

    # Calculate the constant for the local Lipschitz continuity (robustness score), and some related statistics
    amplification = np.linalg.norm(y0 - y_perturbations, axis=1) / np.linalg.norm(x0 - x_perturbations, axis=1)
    L = np.max(amplification)

    x_worst_case = x_perturbations[np.argmax(amplification)]

    if x_range:
        for i, (x_min, x_max) in enumerate(x_range):
            x_worst_case[i] = scale_inverse(x_perturbations[np.argmax(amplification)][i], x_min, x_max)

    return L, x_worst_case


def get_perturbations(
    func:Callable[[ArrayLike], ArrayLike],
    x0: ArrayLike,
    input_space:List[Tuple]=None,
    input_perturbations:ArrayLike=None,
    num_perturbations:int=100,
    pct_perturbation_factor:float=0.01,
    *args, **kwargs
    ) -> Tuple[ArrayLike, ArrayLike]:
    '''
    Score describing the robustness of an explanation model in terms of
    its constant L in the definition of “local Lipschitz continuity”.

    :func: A callable that accepts numpy arrays with shape (P, num_features), and return a numpy array with shape (P, num_labels)
    :x0: Point of interest for where to calculate a robustness score
    :input_space: Min and max values for each input, given as a list of tuples like this `[(min_x1, max_x1), ...]`
    :input_perturbations: An alternative to the input_space where the user can define a custom space of pre-sampled input perturbations.
                        If input_space is given, this parameter will be ignored.
                        If input_perturbations is given, N will be the length of input_perturbations along the first axis
                        Note that the perturbations values should be given with the same unit as the original input features (not as deviations from x0)
    :num_perturbations: Number of permutations
    :pct_perturbation_factor: Size of permutations in percentage of range
    
    :return (x0, y0, x_perturbations, y_perturbations): x0 is the same as the input.
                                                        y0=func(x0) has shape (1,M).
                                                        Perturbed inputs and outputs have shapes(P,N) and (P,M) respectively
    '''
    # Original prediction
    if x0.ndim == 1: x0 = x0.reshape(1,-1)
    y0 = func(x0)

    # Generate N perturbations around x0
    if input_space is not None:
        # Convert input_space to numpy array of shape (num_input, 2)
        input_space = np.array(input_space).T

        # Calculate perturbation size for each input
        max_perturbation = pct_perturbation_factor * (np.max(input_space, axis=0) - np.min(input_space, axis=0))

        # Generate N perturbations away from x0/"the point of interest"
        delta = np.random.uniform(low=-max_perturbation, high=max_perturbation, size=(num_perturbations, input_space.shape[1]))

        # Calculate perturbations
        x_perturbations = x0 + delta

    # Use the perturbations given in input_perturbations
    elif input_perturbations is not None:
        x_perturbations = input_perturbations

    else:
        ValueError("Neither input_space nor input_perturbations was specified")

    # Calculate output
    y_perturbations = func(x_perturbations, *args, **kwargs)

    return x0, y0, x_perturbations, y_perturbations


def explainer_func_wrapper(predict_fn:object, explainer_handle:object, explainer:Literal['lime']='lime', *args, **kwargs) -> Callable[[ArrayLike], ArrayLike]:
    ''' Wrapper for `calc_explanations` that passes the ml_model and explainer instance
    :predict_fn: An method for making predictions. `model.precit()` for regressors and `model.predict_proba()` for classifisers. Must return a format supported by lime
    :explainer_handle: An instantiated explainer object like LIME's `lime_tabular.LimeTabularExplainer`
    :explainer: A string specifying what type of explainer handle was passed, e.g. 'lime'
    :args,kwargs: Parameters passed on to the explainer handle
    :return: A function that accepts array-like structures of shape (P,N) and return array-like structures of shape (P,M)
    '''
    if explainer == 'lime':
        def func(x_inputs:ArrayLike):
            ''' This function wraps the LIME explanation model in order to
            remove the categorical features from the datapoint. The reason
            this is that previous research reveiled th  at including the
            categorical features made the LIME explanations unstable.
            :x_inputs: An array-like structure of shape (P,N), where P is instances to explain and N is number of inputs to the predict_fn
            '''
            if x_inputs.ndim == 1: x_inputs = x_inputs.reshape(1,-1)
            
            exp_list = []
            for x in x_inputs:
                exp_list.append(explainer_handle.explain_instance(
                    data_row = x,
                    predict_fn = predict_fn,
                    *args, **kwargs
                    ))
            return np.array([[value for feature, value in exp.as_list()] for exp in exp_list])

    elif explainer == 'shap':
        raise NotImplemented

    else:
        raise NotImplemented
    
    return func