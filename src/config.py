import yaml

from src.paths import CONFIG_PATH


def load_config():
    """
    Legge il file config/config.yaml
    e restituisce la configurazione
    sotto forma di dizionario Python.
    """

    with open(
        CONFIG_PATH,
        mode="r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    return config