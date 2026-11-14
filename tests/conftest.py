import copy

import pytest

from gasolina_mlops import layers, simulate
from gasolina_mlops.config import layer_uri, load_config
from gasolina_mlops.train import train


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    c = copy.deepcopy(load_config("configs/config.yaml"))
    c["storage"]["base_uri"] = str(tmp_path_factory.mktemp("data"))
    return c


@pytest.fixture(scope="session")
def fotos(cfg):
    return simulate.generate_photos(cfg, cfg["seed"])


@pytest.fixture(scope="session")
def cusn(cfg, fotos):
    simulate.write_photos(fotos, layer_uri(cfg, "raw"))
    return layers.to_cusn(layer_uri(cfg, "raw"))


@pytest.fixture(scope="session")
def ag(cusn):
    return layers.to_ag(cusn)


@pytest.fixture(scope="session")
def au(ag):
    return layers.to_au(ag)


@pytest.fixture(scope="session")
def trained(au, cfg, tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("models")
    return model_dir, train(au["modelo"], cfg, model_dir)
