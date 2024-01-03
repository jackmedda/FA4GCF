import os
import re
import yaml

from recbole.config import Config as Recbole_Config

from fa4gcf.utils import get_model


class Config(Recbole_Config):

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

    def update_base_explainer(self, explainer_config_file=None):
        current_file = os.path.dirname(os.path.realpath(__file__))
        base_explainer_config_file = os.path.join(
            current_file, os.pardir, os.pardir, os.pardir, "config", "explainer", "base_explainer.yaml"
        )
        with open(base_explainer_config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.load(f.read(), Loader=self.yaml_loader)

        if explainer_config_file is not None:
            if os.path.splitext(explainer_config_file)[-1] == '.yaml':
                with open(explainer_config_file, 'r', encoding='utf-8') as f:
                    exp_config_dict = yaml.load(f.read(), Loader=self.yaml_loader)
            else:
                with open(explainer_config_file, 'rb') as f:
                    exp_config_dict = pickle.load(f).final_config_dict

            config_dict.update(exp_config_dict)

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
