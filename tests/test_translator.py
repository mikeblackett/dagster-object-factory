import pytest
import xarray as xr
import xclim.indicators.atmos as indicators

from dagster_object_factory import Layer, LayerDep
from dagster_object_factory.integrations.xclim import (
    DagsterXclimIndicatorTranslator,
)

JETSTREAM_OUTPUTS = ["jetlat", "jetstr"]


def _translator(output_names, key_prefix=("indicators",), layer_deps=None):
    layer = Layer(
        name="indicators",
        key_prefix=key_prefix,
        output_names=frozenset(output_names),
        python_type=xr.DataArray,
        layer_deps=layer_deps
        if layer_deps is not None
        else (LayerDep(layer=_variables_layer()),),
    )
    return DagsterXclimIndicatorTranslator(layer=layer)


def _variables_layer():
    return Layer(
        name="variables",
        key_prefix=("variables",),
        output_names=frozenset({"tas", "tasmin", "tasmax", "ua"}),
        python_type=xr.DataArray,
    )


def test_single_output_names_and_keys():
    translation = _translator(["frost_days"])(indicators.frost_days)
    assert translation.output_names == ("frost_days",)
    assert [key.to_user_string() for key in translation.keys] == [
        "indicators/frost_days"
    ]


def test_multi_output_names_and_keys():
    translation = _translator(JETSTREAM_OUTPUTS)(indicators.jetstream_metric_woollings)
    assert translation.output_names == tuple(JETSTREAM_OUTPUTS)
    assert [key.to_user_string() for key in translation.keys] == [
        f"indicators/{name}" for name in JETSTREAM_OUTPUTS
    ]


def test_asset_outs_single():
    translation = _translator(["frost_days"])(indicators.frost_days)
    outs = translation.get_asset_outs()
    assert list(outs) == ["frost_days"]
    out = outs["frost_days"]
    assert out.key is not None
    assert out.key.to_user_string() == "indicators/frost_days"
    assert out.dagster_type is xr.DataArray


def test_asset_outs_multi():
    translation = _translator(JETSTREAM_OUTPUTS)(indicators.jetstream_metric_woollings)
    outs = translation.get_asset_outs()
    assert set(outs) == set(JETSTREAM_OUTPUTS)
    jetstr = outs["jetstr"]
    assert jetstr.key is not None
    assert jetstr.key.to_user_string() == "indicators/jetstr"


def test_asset_spec_carries_deps():
    translation = _translator(["frost_days"])(indicators.frost_days)
    spec = translation.get_asset_spec("frost_days")
    assert spec.key.to_user_string() == "indicators/frost_days"
    deps = [dep.asset_key.to_user_string() for dep in (spec.deps or [])]
    assert deps == ["variables/tasmin"]


def test_undeclared_output_raises():
    # Layer declares only frost_days, but the producer emits cffwis outputs.
    translator = _translator(["frost_days"])
    with pytest.raises(ValueError, match="not a declared output"):
        translator(indicators.cffwis_indices)


def test_multi_output_descriptions_from_cf_attrs():
    obj = indicators.jetstream_metric_woollings
    translation = _translator(JETSTREAM_OUTPUTS)(obj)
    cf_descriptions = {
        attrs["var_name"]: attrs["description"] for attrs in obj.cf_attrs
    }
    outs = translation.get_asset_outs()
    for name in JETSTREAM_OUTPUTS:
        assert outs[name].description == cf_descriptions[name]


def test_single_output_description_broadcast():
    obj = indicators.frost_days
    translation = _translator(["frost_days"])(obj)
    assert translation.description is not None
    assert obj.title in translation.description
    assert (
        translation.get_asset_outs()["frost_days"].description
        == translation.description
    )
