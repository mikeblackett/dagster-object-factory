import pytest
import xarray as xr
import xclim.indicators.atmos as indicators

from dagster_object_factory import Layer, LayerDep
from dagster_object_factory.integrations.xclim import (
    DagsterXclimIndicatorTranslator,
    XclimIndicatorFactory,
)

CFFWIS_OUTPUTS = ["dc", "dmc", "ffmc", "isi", "bui", "fwi"]


def _variables_layer():
    return Layer(
        name="variables",
        key_prefix=("variables",),
        output_names=frozenset({"tas", "pr", "sfcWind", "hurs", "lat", "tasmin"}),
        python_type=xr.DataArray,
    )


def _factory(obj, output_names):
    layer = Layer(
        name="indicators",
        key_prefix=("t",),
        output_names=frozenset(output_names),
        python_type=xr.DataArray,
        layer_deps=(LayerDep(layer=_variables_layer()),),
    )
    translator = DagsterXclimIndicatorTranslator(layer=layer)
    return XclimIndicatorFactory(objects=[obj], translator=translator)


def test_single_output_is_wrapped():
    factory = _factory(indicators.frost_days, ["frost_days"])
    translation = factory.translator(indicators.frost_days)
    value = xr.DataArray([1.0])
    pairs = list(translation.iter_output_values(value))
    assert [name for name, _ in pairs] == ["frost_days"]
    assert pairs[0][1] is value
    assert (
        translation.keys_by_output_name["frost_days"].to_user_string() == "t/frost_days"
    )


def test_multi_output_is_decomposed_and_aligned():
    factory = _factory(indicators.cffwis_indices, CFFWIS_OUTPUTS)
    translation = factory.translator(indicators.cffwis_indices)
    values = tuple(xr.DataArray([float(i)]) for i in range(len(CFFWIS_OUTPUTS)))
    pairs = list(translation.iter_output_values(values))
    assert [name for name, _ in pairs] == CFFWIS_OUTPUTS
    for index, (_, value) in enumerate(pairs):
        assert value is values[index]


def test_length_mismatch_raises():
    factory = _factory(indicators.cffwis_indices, CFFWIS_OUTPUTS)
    translation = factory.translator(indicators.cffwis_indices)
    short = tuple(xr.DataArray([float(i)]) for i in range(3))
    with pytest.raises(ValueError):
        list(translation.iter_output_values(short))
