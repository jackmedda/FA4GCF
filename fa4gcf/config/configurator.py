import os
import re
import yaml
import pickle

from recbole.utils import ModelType
from recbole.config import Config as Recbole_Config

from fa4gcf.utils import get_model


class Config(Recbole_Config):

    DONT_LOAD_MODEL_PARAMS = "DONT_LOAD_MODEL_PARAMS"

    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):
        super(Config, self).__init__(
            model=model,
            dataset=dataset,
            config_file_list=config_file_list,
            config_dict=config_dict
        )

        svd_gcn_exception = model == "SVD_GCN" and not self.final_config_dict.get("parametric", True)
        if self["MODEL_TYPE"] == ModelType.TRADITIONAL or svd_gcn_exception:
            self["train_neg_sample_args"]["sample_num"] = 1

    def _get_model_and_dataset(self, model, dataset):
        if model is None:
            try:
                model = self.external_config_dict["model"]
            except KeyError:
                raise KeyError(
                    "model need to be specified in at least one of the these ways: "
                    "[model variable, config file, config dict, command line] "
                )
        if not isinstance(model, str):
            final_model_class = model
            final_model = model.__name__
        elif model == self.DONT_LOAD_MODEL_PARAMS:
            final_model = model
            final_model_class = get_model("LightGCN")  # used just to get ModelType = GENERAL
        else:
            final_model = model
            final_model_class = get_model(final_model)

        if dataset is None:
            try:
                final_dataset = self.external_config_dict["dataset"]
            except KeyError:
                raise KeyError(
                    "dataset need to be specified in at least one of the these ways: "
                    "[dataset variable, config file, config dict, command line] "
                )
        else:
            final_dataset = dataset

        return final_model, final_model_class, final_dataset

    def update_base_perturb_data(self, perturbation_config_file=None):
        current_file = os.path.dirname(os.path.realpath(__file__))
        base_perturbation_config_file = os.path.join(
            current_file, os.pardir, os.pardir, "config", "perturbation", "base_perturbation.yaml"
        )
        with open(base_perturbation_config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.load(f.read(), Loader=self.yaml_loader)

        if perturbation_config_file is not None:
            if os.path.splitext(perturbation_config_file)[-1] == '.yaml':
                with open(perturbation_config_file, 'r', encoding='utf-8') as f:
                    pert_config_dict = yaml.load(f.read(), Loader=self.yaml_loader)
            else:
                with open(perturbation_config_file, 'rb') as f:
                    pert_config_dict = pickle.load(f).final_config_dict

            config_dict.update(pert_config_dict)

        return config_dict

    @staticmethod
    def _build_yaml_loader():
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader
