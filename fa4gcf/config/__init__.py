from fa4gcf.config.configurator import Config
from fa4gcf.evaluation.register import metric_types as fa4gcf_metric_types, smaller_metrics as fa4gcf_smaller_metrics

import recbole.config.configurator as recbole_config_module

# extends the set of supported metrics
recbole_config_module.metric_types.update(fa4gcf_metric_types)
recbole_config_module.smaller_metrics.extend(fa4gcf_smaller_metrics)
