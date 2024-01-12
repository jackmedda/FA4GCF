from functools import partial

import numpy as np
from recbole.trainer import HyperTuning as RecboleHyperTuning


def exhaustive_search(new_ids, domain, trials, seed, nbMaxSucessiveFailures=1000):
    r"""This is for exhaustive search in HyperTuning."""
    from hyperopt import pyll
    from hyperopt.base import miscs_update_idxs_vals

    # Build a hash set for previous trials
    hashset = set(
        [
            hash(
                frozenset(
                    [
                        (key, value[0]) if len(value) > 0 else ((key, None))
                        for key, value in trial["misc"]["vals"].items()
                    ]
                )
            )
            for trial in trials.trials
        ]
    )

    rng = np.random.default_rng(seed)
    rval = []
    for _, new_id in enumerate(new_ids):
        newSample = False
        nbSucessiveFailures = 0
        while not newSample:
            # -- sample new specs, idxs, vals
            idxs, vals = pyll.rec_eval(
                domain.s_idxs_vals,
                memo={
                    domain.s_new_ids: [new_id],
                    domain.s_rng: rng,
                },
            )
            new_result = domain.new_result()
            new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
            miscs_update_idxs_vals([new_misc], idxs, vals)

            # Compare with previous hashes
            h = hash(
                frozenset(
                    [
                        (key, value[0]) if len(value) > 0 else ((key, None))
                        for key, value in vals.items()
                    ]
                )
            )
            if h not in hashset:
                newSample = True
            else:
                # Duplicated sample, ignore
                nbSucessiveFailures += 1

            if nbSucessiveFailures > nbMaxSucessiveFailures:
                # No more samples to produce
                return []

        rval.extend(trials.new_trial_docs([new_id], [None], [new_result], [new_misc]))
    return rval


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
        if algo == "exhaustive":
            algo = partial(exhaustive_search, nbMaxSucessiveFailures=1000)

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
        import hyperopt
        try:
            return super(HyperTuning, self).trial(params)
        except Exception as e:
            if self.ignore_errors:
                return {"loss": np.nan, "status": hyperopt.STATUS_FAIL}
            else:
                raise e
