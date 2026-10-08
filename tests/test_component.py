import dagster as dg
import pytest as pt
import xarray as xr
import xclim.core.indicator as xc
import xclim.indicators.atmos as indicators
from xclim.indicators import land

from dagster_object_factory import Layer, LayerDep
from dagster_object_factory.integrations.xclim import (
    DagsterXclimIndicatorTranslator,
    XclimIndicatorFactory,
)
from dagster_object_factory.integrations.xclim.components import (
    XclimResamplingIndicatorFactory,
)
from dagster_object_factory.integrations.xclim.partitions import (
    ResamplingPartitionsDefinition,
)


def _variables_layer():
    return Layer(
        name="variables",
        key_prefix=("variables",),
        output_names=frozenset({"tas", "tasmin", "tasmax", "q"}),
        python_type=xr.DataArray,
    )


@pt.fixture()
def frost_days_injected_freq() -> xc.ResamplingIndicator:
    # TODO: fixture is kinda pointless here, because the custom indicator
    #   will be added to xclim's registry and persist for the session.
    return xc.registry["FROST_DAYS"](
        identifier="frost_days_test",
        parameters={"freq": "YS-JUN"},  # We inject the freq arg.
    )


def _factory(
    obj,
    output_names,
    key_prefix,
    partitions_def=None,
    injected_kwargs=None,
    factory_class=XclimIndicatorFactory,
):
    layer = Layer(
        name="indicators",
        key_prefix=key_prefix,
        output_names=frozenset(output_names),
        python_type=xr.DataArray,
        layer_deps=(LayerDep(layer=_variables_layer()),),
        partitions_def=partitions_def,
    )
    translator = DagsterXclimIndicatorTranslator(layer=layer)
    return factory_class(
        objects=[obj],
        translator=translator,
        injected_kwargs=injected_kwargs or {},
    )


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


def test_injecting_already_injected_parameter_raises(frost_days_injected_freq):
    """Injecting a parameter that has already been injected by xclim should raise an error"""
    factory = _factory(
        indicators.frost_days,
        ["frost_days_test"],
        ("indicators",),
        injected_kwargs={"freq": "MS"},
    )
    assert "freq" in frost_days_injected_freq.injected_parameters
    assert "freq" not in frost_days_injected_freq.parameters

    with pt.raises(dg.DagsterInvalidDefinitionError):
        factory.make_asset(
            context=dg.ComponentTree.for_test().load_context,
            obj=frost_days_injected_freq,
        )


def test_injecting_unknown_parameter_raises():
    """Injecting a parameter that is not an indicator parameter should raise an error"""
    factory = _factory(
        indicators.frost_days,
        ["frost_days"],
        ("indicators",),
        injected_kwargs={"sausages": "yes"},
    )

    with pt.raises(dg.DagsterInvalidDefinitionError):
        factory.make_asset(
            context=dg.ComponentTree.for_test().load_context,
            obj=indicators.frost_days,
        )


def test_freq_partition_key_is_resolved():
    factory = _factory(
        indicators.frost_days,
        ["frost_days"],
        ("indicators",),
        partitions_def=ResamplingPartitionsDefinition(["MS"]),
        factory_class=XclimResamplingIndicatorFactory,
    )
    kwargs = factory.resolve_execution_kwargs(
        context=dg.build_asset_context(partition_key="MS"),
        obj=indicators.frost_days,
    )

    assert "freq" in kwargs
    assert kwargs["freq"] == "MS"


def test_freq_multi_partition_key_is_resolved():
    factory = _factory(
        indicators.frost_days,
        ["frost_days"],
        ("indicators",),
        partitions_def=dg.MultiPartitionsDefinition(
            {
                "freq": ResamplingPartitionsDefinition(["MS"]),
                "other": dg.StaticPartitionsDefinition(["a"]),
            }
        ),
        factory_class=XclimResamplingIndicatorFactory,
    )
    kwargs = factory.resolve_execution_kwargs(
        context=dg.build_asset_context(
            partition_key=dg.MultiPartitionKey({"freq": "MS", "other": "a"})
        ),
        obj=indicators.frost_days,
    )

    assert "freq" in kwargs
    assert kwargs["freq"] == "MS"


def test_invalid_injected_freq_raises():
    factory = _factory(
        indicators.frost_days,
        ["frost_days"],
        ("indicators",),
        factory_class=XclimResamplingIndicatorFactory,
        injected_kwargs={"freq": "tuna"},
    )
    with pt.raises(dg.DagsterInvalidDefinitionError):
        factory.make_asset(
            context=dg.ComponentTree.for_test().load_context,
            obj=indicators.frost_days,
        )


def test_invalid_freq_partition_key_raises():
    factory = _factory(
        indicators.frost_days,
        ["frost_days"],
        ("indicators",),
        partitions_def=ResamplingPartitionsDefinition(["MS", "QS-NOV"]),
        factory_class=XclimResamplingIndicatorFactory,
    )
    with pt.raises(dg.DagsterInvalidDefinitionError):
        factory.resolve_execution_kwargs(
            context=dg.build_asset_context(partition_key="eggs"),
            obj=indicators.frost_days,
        )


def test_injected_freq_without_partitions_is_used():
    """A resampling factory with an injected freq and no partitions uses the injected freq."""
    factory = _factory(
        indicators.frost_days,
        ["frost_days"],
        ("indicators",),
        factory_class=XclimResamplingIndicatorFactory,
        injected_kwargs={"freq": "MS"},
    )
    kwargs = factory.resolve_injected_kwargs(
        context=dg.ComponentTree.for_test().load_context,
        obj=indicators.frost_days,
    )
    assert kwargs == {"freq": "MS"}


def test_injected_freq_conflicts_with_resampling_partitions_raises():
    """Setting an injected freq on a layer partitioned by a ResamplingPartitionsDefinition is an error."""
    factory = _factory(
        indicators.frost_days,
        ["frost_days"],
        ("indicators",),
        partitions_def=ResamplingPartitionsDefinition(["MS"]),
        factory_class=XclimResamplingIndicatorFactory,
        injected_kwargs={"freq": "MS"},
    )
    with pt.raises(dg.DagsterInvalidDefinitionError):
        factory.make_asset(
            context=dg.ComponentTree.for_test().load_context,
            obj=indicators.frost_days,
        )


def test_injected_freq_conflicts_with_multi_resampling_partition_raises():
    """Setting an injected freq on a layer with a resampling partition dimension is an error."""
    factory = _factory(
        indicators.frost_days,
        ["frost_days"],
        ("indicators",),
        partitions_def=dg.MultiPartitionsDefinition(
            {
                "freq": ResamplingPartitionsDefinition(["MS"]),
                "other": dg.StaticPartitionsDefinition(["a"]),
            }
        ),
        factory_class=XclimResamplingIndicatorFactory,
        injected_kwargs={"freq": "MS"},
    )
    with pt.raises(dg.DagsterInvalidDefinitionError):
        factory.make_asset(
            context=dg.ComponentTree.for_test().load_context,
            obj=indicators.frost_days,
        )
