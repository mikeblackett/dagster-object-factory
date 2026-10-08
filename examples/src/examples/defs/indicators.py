import dagster as dg
import xarray as xr
import xclim.indicators.atmos as indicators

from dagster_object_factory import Layer, LayerDep
from dagster_object_factory.integrations.xclim import (
    DagsterXclimIndicatorTranslator,
    ResamplingPartitionsDefinition,
    XclimIndicatorFactory,
    XclimResamplingIndicatorFactory,
)

from .variables import layer as variable_layer

single_output = Layer(
    name="single_output",
    description="xclim resampling indicators computed over a single frequency.",
    key_prefix=("indicators", "single"),
    output_names=frozenset({"frost_days"}),
    python_type=xr.DataArray,
    layer_deps=(LayerDep(layer=variable_layer),),
)

partitioned_single_output = Layer(
    name="partitioned_single_output",
    description="xclim resampling indicators partitioned over frequency.",
    key_prefix=("indicators", "single", "partitioned"),
    output_names=frozenset({"tropical_nights"}),
    python_type=xr.DataArray,
    layer_deps=(LayerDep(layer=variable_layer),),
    partitions_def=ResamplingPartitionsDefinition(["MS", "QS-NOV", "YS"]),
)

multi = Layer(
    name="indicators",
    key_prefix=("indicators", "multi"),
    output_names=frozenset({"jetlat", "jetstr"}),
    python_type=xr.DataArray,
    layer_deps=(LayerDep(layer=variable_layer),),
)


@dg.component_instance
def single_output_assets(context: dg.ComponentLoadContext):
    translator = DagsterXclimIndicatorTranslator(layer=single_output)
    return XclimResamplingIndicatorFactory(
        objects=[indicators.frost_days],
        translator=translator,
        injected_kwargs={"freq": "MS"},
    )


@dg.component_instance
def partitioned_single_output_assets(context: dg.ComponentLoadContext):
    translator = DagsterXclimIndicatorTranslator(layer=partitioned_single_output)
    return XclimResamplingIndicatorFactory(
        objects=[indicators.tropical_nights],
        translator=translator,
    )


@dg.component_instance
def multi_indicator_assets(context: dg.ComponentLoadContext):
    translator = DagsterXclimIndicatorTranslator(layer=multi)
    return XclimIndicatorFactory(
        objects=[indicators.jetstream_metric_woollings],
        translator=translator,
    )
