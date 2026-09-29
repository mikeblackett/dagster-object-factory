from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, cast

import dagster as dg
import xarray as xr
import xclim.core.calendar as xcal
import xclim.core.indicator as xc

from dagster_object_factory.component import (
    ObjectFactoryComponent,
)
from dagster_object_factory.integrations.xclim.partitions import (
    ResamplingPartitionsDefinition,
)
from dagster_object_factory.integrations.xclim.types import Chunks

XCLIM_FREQUENCY_KEYWORD: Final = "freq"


@dataclass(frozen=True, kw_only=True)
class XclimIndicatorFactory[T: xc.Indicator](ObjectFactoryComponent[T, xr.DataArray]):
    injected_parameters: Mapping[str, Any] = field(default_factory=dict)

    def execute(
        self,
        context: dg.AssetExecutionContext,
        obj: T,
        ins: Mapping[str, xr.DataArray],
        **kwargs,
    ) -> xr.DataArray:
        result = cast(xr.DataArray, obj(**ins, **kwargs))
        result = result.chunk(self.get_chunks(result))
        return result

    def get_metadata(
        self, context: dg.AssetExecutionContext, result: xr.DataArray
    ) -> dict[str, dg.MetadataValue]:
        return {
            **super().get_metadata(context, result),
            "nbytes": dg.MetadataValue.int(result.nbytes),
        }

    def get_chunks(self, obj: xr.DataArray) -> Chunks:
        return None


@dataclass(frozen=True, kw_only=True)
class XclimResamplingIndicatorFactory(XclimIndicatorFactory[xc.ResamplingIndicator]):
    def _resolve_freq(
        self, context: dg.AssetExecutionContext, obj: xc.ResamplingIndicator
    ) -> str:
        if "freq" in self.injected_parameters:
            freq = self.injected_parameters["freq"]
        elif isinstance(
            self.translator.layer.partitions_def, ResamplingPartitionsDefinition
        ):
            freq = self.get_partition_key(context, "freq")
        else:
            raise

        allowed_periods = obj.allowed_periods
        if allowed_periods is None:
            return freq
        period = xcal.parse_offset(freq)[1]
        if period not in allowed_periods:
            raise
        return freq

    def resolve_execution_kwargs(
        self, context: dg.AssetExecutionContext, obj: xc.ResamplingIndicator
    ) -> dict[str, Any]:
        return {
            XCLIM_FREQUENCY_KEYWORD: self.get_partition_key(
                context, self._resolve_freq(context, obj)
            )
        }
