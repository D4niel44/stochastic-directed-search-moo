# =========================================================================================================
# Imports
# =========================================================================================================
from src.sds.core.corrector import DeltaCorrector, RankCorrector, ProjectionCorrector
from src.sds.core.predictor import StepAdjust, NoAdjustmentPredictors, LimitsPredictors
from src.sds.core.stepsize import Dominance, AngleBisection, Armijo, WeightedDominance, Angle
from src.sds.core.termination import MaxIter, Tol, NullTermination

# =========================================================================================================
# Generic
# =========================================================================================================


def get_from_list(l, name, args, kwargs):
    i = None

    for k, e in enumerate(l):
        if e[0] == name:
            i = k
            break

    if i is None:
        for k, e in enumerate(l):
            if re.match(e[0], name):
                i = k
                break

    if i is not None:

        if len(l[i]) == 2:
            name, clazz = l[i]

        elif len(l[i]) == 3:
            name, clazz, default_kwargs = l[i]

            # overwrite the default if provided
            for key, val in kwargs.items():
                default_kwargs[key] = val
            kwargs = default_kwargs

        return clazz(*args, **kwargs)
    else:
        raise Exception("Object '%s' for not found in %s" % (name, [e[0] for e in l]))


# =========================================================================================================
# T Functions
# =========================================================================================================

def get_tfun_options():
    SAMPLING = [
        ("angle", Angle),
        ("dominance", Dominance),
        ("angle_bisection", AngleBisection),
        ("armijo", Armijo),
        ("weighted_dominance", WeightedDominance),
    ]

    return SAMPLING


def get_tfun(name, *args, d={}, **kwargs):
    return get_from_list(get_tfun_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Correctors
# =========================================================================================================

def get_corrector_options():
    SAMPLING = [
        ("projection", ProjectionCorrector),
        ("rank", RankCorrector),
        ("delta", DeltaCorrector)
    ]

    return SAMPLING


def get_corrector(name, *args, d={}, **kwargs):
    return get_from_list(get_corrector_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Termination
# =========================================================================================================

def get_termination_options():
    SAMPLING = [
        ("none", NullTermination),
        ("n_iter", MaxIter),
        ("tol", Tol)
    ]

    return SAMPLING


def get_cont_termination(name, *args, d={}, **kwargs):
    return get_from_list(get_termination_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Correctors
# =========================================================================================================

def get_predictor_options():
    SAMPLING = [
        ("step_adjust", StepAdjust),
        ("no_adjustment", NoAdjustmentPredictors),
        ("limit", LimitsPredictors),
    ]

    return SAMPLING


def get_predictor(name, *args, d={}, **kwargs):
    return get_from_list(get_predictor_options(), name, args, {**d, **kwargs})
