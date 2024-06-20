import os
import sys
import argparse

import torch
current_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file, os.pardir))

import fa4gcf.utils as utils
from fa4gcf.config import Config


if __name__ == "__main__":
    """It works only when called from outside of the scripts folder as a script (not as a module)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', '-mf', required=True)
    parser.add_argument('--sensitive_attribute', '-sa', required=True)
    parser.add_argument('--gpu_id', default=1)
    args, _ = parser.parse_known_args()

    consumer_group_map = {
        'gender': {'M': 'M', 'F': 'F'},
        'age': {'M': 'Y', 'F': 'O'},
        'user_wide_zone': {'M': 'America', 'F': 'Other'}
    }

    group_name_map = {
        "M": "Males",
        "F": "Females",
        "Y": "Younger",
        "O": "Older",
        "America": "America",
        "Other": "Other"
    }

    checkpoint = torch.load(args.model_file)

    dset, mod = checkpoint["config"]["dataset"], checkpoint["config"]["model"]
    s_attr = args.sensitive_attribute

    config = Config(
        model=Config.DONT_LOAD_MODEL_PARAMS,
        dataset=dset,
        config_file_list=[
            os.path.join(current_file, '..', 'config', 'base_config.yaml'),
            os.path.join(current_file, '..', 'config', 'dataset', f'{dset}.yaml')
        ],
        config_dict={"gpu_id": args.gpu_id, "sensitive_attribute": args.sensitive_attribute}
    )
    config["model"] = mod
    config, model, dataset, train_data, valid_data, test_data = utils.load_data_and_model(
        args.model_file,
        config.final_config_dict  # it could contain updated information, e.g., it should load the user_feat
    )

    trainer = utils.get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    print(
        "Dataset:", config["dataset"],
        "Model:", config["model"],
        "Sensitive attribute:", config["sensitive_attribute"]
    )
    for split, eval_data in zip(['Test', 'Valid'], [test_data, valid_data]):
        result = trainer.evaluate(eval_data, load_best_model=True, model_file=args.model_file)
        print(split)
        print(result, end='\n\n')
