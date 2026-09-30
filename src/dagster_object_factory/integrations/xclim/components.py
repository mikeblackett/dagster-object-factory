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
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
        result = obj(**ins, **kwargs)
        if isinstance(result, tuple):
            chunked = tuple(data.chunk(self.get_chunks(data)) for data in result)
            return cast(tuple[xr.DataArray, ...], chunked)
        data = cast(xr.DataArray, result)
        result = data.chunk(self.get_chunks(data))
        return result

    def get_metadata(
        self, context: dg.AssetExecutionContext, value: xr.DataArray
    ) -> dict[str, dg.MetadataValue]:
        return {
            **super().get_metadata(context, value),
            "nbytes": dg.MetadataValue.int(value.nbytes),
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
            raise dg.DagsterInvalidDefinitionError(
                f"{obj.identifier}: no frequency configured: set 'freq' in "
                "injected_parameters or use a ResamplingPartitionsDefinition."
            )

        allowed_periods = obj.allowed_periods
        if allowed_periods is None:
            return freq
        period = xcal.parse_offset(freq)[1]
        if period not in allowed_periods:
            raise ValueError(
                f"{obj.identifier}: frequency {freq!r} has period {period!r}, "
                f"which is not allowed (allowed: {allowed_periods!r})."
            )
        return freq

    def resolve_execution_kwargs(
        self, context: dg.AssetExecutionContext, obj: xc.ResamplingIndicator
    ) -> dict[str, Any]:
        return {
            XCLIM_FREQUENCY_KEYWORD: self.get_partition_key(
                context, self._resolve_freq(context, obj)
            )
        }
