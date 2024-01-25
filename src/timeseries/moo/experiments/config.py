from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA


supported_moeas = [
    NSGA2,
    NSGA3,
    MOEAD,
    SMSEMOA,
]

supported_problem_sizes = [
    'small',
    'medium',
    'large'
]

moea_map = {m.__name__: m for m in supported_moeas}