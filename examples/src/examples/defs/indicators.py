import dagster as dg
import xarray as xr
import xclim.indicators.atmos as indicators

from dagster_object_factory import Layer, LayerDependency
from dagster_object_factory.integrations.xclim import (
    DagsterXclimIndicatorTranslator,
    ResamplingPartitionsDefinition,
    XclimResamplingIndicatorFactory,
)

from .variables import layer as variable_layer

layer = Layer(
    name="indicators",
    key_prefix=("indicators",),
    output_names=frozenset({"frost_days"}),
    python_type=xr.DataArray,
    sources=(LayerDependency(layer=variable_layer),),
    partitions_def=ResamplingPartitionsDefinition(["MS", "QS-NOV", "YS"]),
)


@dg.component_instance
def source_indicator_assets(context: dg.ComponentLoadContext):
    translator = DagsterXclimIndicatorTranslator(layer=layer)
    return XclimResamplingIndicatorFactory(
        objects=[getattr(indicators, i) for i in layer.output_names],
        translator=translator,
    )
