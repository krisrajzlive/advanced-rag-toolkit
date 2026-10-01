import pytest
from qdrant_client import models

from src.indexing.hnsw import HnswParams, hnsw_config
from src.indexing.quantization import quantization_config, search_params
from src.tenancy.auth import TenantContext
from src.tenancy.tenant_rag import tenant_filters


def test_quantization_configs_are_native_qdrant_models():
    assert quantization_config("none") is None
    assert isinstance(quantization_config("scalar"), models.ScalarQuantization)
    assert isinstance(quantization_config("binary"), models.BinaryQuantization)
    assert isinstance(quantization_config("product"), models.ProductQuantization)
    with pytest.raises(ValueError):
        quantization_config("bogus")


def test_search_params_rescore_and_ef():
    p = search_params("scalar", oversampling=2.0, rescore=True, hnsw_ef=64)
    assert p.hnsw_ef == 64 and p.quantization.rescore and p.quantization.oversampling == 2.0
    assert search_params("none").quantization is None


def test_hnsw_config_maps_params():
    cfg = hnsw_config(HnswParams(m=32, ef_construct=200, payload_m=16))
    assert (cfg.m, cfg.ef_construct, cfg.payload_m) == (32, 200, 16)


def test_tenant_filter_restricts_to_verified_tenants():
    f = tenant_filters(TenantContext("alice", ("acme",))).filters[0]
    assert f.key == "tenant_id" and f.value == ["acme"]
