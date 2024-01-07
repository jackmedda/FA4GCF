import hyperopt
from recbole.trainer import HyperTuning as RecboleHyperTuning


class HyperTuning(RecboleHyperTuning):

    def __init__(
        self,
        objective_function,
        model,
        space=None,
        params_file=None,
        params_dict=None,
        fixed_config_file_list=None,
        display_file=None,
        ignore_errors=False,
        algo="exhaustive",
        max_evals=100,
        early_stop=10,
    ):
        self.model = model
        self.ignore_errors = ignore_errors
        super(HyperTuning, self).__init__(
            objective_function,
            space=space,
            params_file=params_file,
            params_dict=params_dict,
            fixed_config_file_list=fixed_config_file_list,
            display_file=display_file,
            algo=algo,
            max_evals=max_evals,
            early_stop=early_stop
        )

    def _build_space_from_file(self, file):
        from hyperopt import hp

        space = {}
        common_params = True  # the first parameters are saved because in common with all models
        found_model = False  # all the params for the model are read when it is True
        with open(file, "r") as fp:
            for line in fp:
                para_list = line.strip().split(" ")
                if len(para_list) < 3:
                    if line.startswith("#"):
                        common_params = False
                        found_model = self.model.lower() == line.strip()[1:].lower()
                    continue
                elif not found_model and not common_params:
                    continue

                para_name, para_type, para_value = (
                    para_list[0],
                    para_list[1],
                    "".join(para_list[2:]),
                )
                if para_type == "choice":
                    para_value = eval(para_value)
                    space[para_name] = hp.choice(para_name, para_value)
                elif para_type == "uniform":
                    low, high = para_value.strip().split(",")
                    space[para_name] = hp.uniform(para_name, float(low), float(high))
                elif para_type == "quniform":
                    low, high, q = para_value.strip().split(",")
                    space[para_name] = hp.quniform(
                        para_name, float(low), float(high), float(q)
                    )
                elif para_type == "loguniform":
                    low, high = para_value.strip().split(",")
                    space[para_name] = hp.loguniform(para_name, float(low), float(high))
                else:
                    raise ValueError("Illegal param type [{}]".format(para_type))
        return space

    def trial(self, params):
        try:
            return super(HyperTuning, self).trial(params)
        except Exception as e:
            if self.ignore_errors:
                return {"status": hyperopt.STATUS_FAIL}
            else:
                raise e
