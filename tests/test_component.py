import dagster as dg
import xarray as xr
import xclim.indicators.atmos as indicators
from xclim.indicators import land

from dagster_object_factory import Layer, LayerDependency
from dagster_object_factory.integrations.xclim import (
    DagsterXclimIndicatorTranslator,
    XclimIndicatorFactory,
)


def _variables_layer():
    return Layer(
        name="variables",
        key_prefix=("variables",),
        output_names=frozenset({"tas", "tasmin", "tasmax", "q"}),
        python_type=xr.DataArray,
    )


def _factory(obj, output_names, key_prefix):
    layer = Layer(
        name="indicators",
        key_prefix=key_prefix,
        output_names=frozenset(output_names),
        python_type=xr.DataArray,
        sources=(LayerDependency(layer=_variables_layer()),),
    )
    translator = DagsterXclimIndicatorTranslator(layer=layer)
    return XclimIndicatorFactory(objects=[obj], translator=translator)


def test_make_asset_single_output_keys():
    factory = _factory(indicators.frost_days, ["frost_days"], ("indicators",))
    asset = factory.make_asset(
        context=dg.ComponentTree.for_test().load_context, obj=indicators.frost_days
    )
    assert [k.to_user_string() for k in asset.keys] == ["indicators/frost_days"]


def test_make_asset_multi_output_keys():
    factory = _factory(land.sen_slope, ["sen_slope", "p_value"], ("trends",))
    asset = factory.make_asset(
        context=dg.ComponentTree.for_test().load_context, obj=land.sen_slope
    )
    assert {k.to_user_string() for k in asset.keys} == {
        "trends/sen_slope",
        "trends/p_value",
    }
    translation = factory.translator(land.sen_slope)
    assert asset.node_def.description == translation.description


def test_build_defs_returns_definitions():
    factory = _factory(land.sen_slope, ["sen_slope", "p_value"], ("trends",))
    defs = factory.build_defs(context=dg.ComponentTree.for_test().load_context)
    assert isinstance(defs, dg.Definitions)
