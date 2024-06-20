from .interaction import Interaction

import recbole.data.dataset.dataset as recbole_dataset_module
recbole_dataset_module.Interaction = Interaction  # overwrites the Interaction class used by Recbole with ours

from .dataset import Dataset
from .perturbed_dataset import PerturbedDataset
