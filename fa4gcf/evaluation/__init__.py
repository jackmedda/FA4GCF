from fa4gcf.evaluation.metrics import *
from fa4gcf.evaluation.collector import Collector
from fa4gcf.evaluation.evaluator import Evaluator
from fa4gcf.evaluation.recommender_evaluation import *
from fa4gcf.evaluation.graph_evaluation import *

import recbole.evaluator.collector as recbole_collector_module
from fa4gcf.evaluation.register import Register

recbole_collector_module.Register = Register
