import dagster as dg
import xarray as xr
import xclim.indicators.atmos as indicators

from dagster_object_factory import Layer, LayerDependency
from dagster_object_factory.integrations.xclim import (
    DagsterXclimIndicatorTranslator,
    ResamplingPartitionsDefinition,
    XclimIndicatorFactory,
    XclimResamplingIndicatorFactory,
)

from .variables import layer as variable_layer

single = Layer(
    name="indicators",
    key_prefix=("indicators", "single"),
    output_names=frozenset({"frost_days"}),
    python_type=xr.DataArray,
    sources=(LayerDependency(layer=variable_layer),),
    partitions_def=ResamplingPartitionsDefinition(["MS", "QS-NOV", "YS"]),
)

multi = Layer(
    name="indicators",
    key_prefix=("indicators", "multi"),
    output_names=frozenset({"jetlat", "jetstr"}),
    python_type=xr.DataArray,
    sources=(LayerDependency(layer=variable_layer),),
)


@dg.component_instance
def single_indicator_assets(context: dg.ComponentLoadContext):
    translator = DagsterXclimIndicatorTranslator(layer=single)
    return XclimResamplingIndicatorFactory(
        objects=[indicators.frost_days],
        translator=translator,
    )


@dg.component_instance
def multi_indicator_assets(context: dg.ComponentLoadContext):
    translator = DagsterXclimIndicatorTranslator(layer=multi)
    return XclimIndicatorFactory(
        objects=[indicators.jetstream_metric_woollings],
        translator=translator,
    )
