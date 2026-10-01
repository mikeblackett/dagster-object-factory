"""Components that execute xclim indicators as dagster assets."""

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
    """An ObjectFactoryComponent that executes xclim indicators.

    Args:
        injected_parameters: Extra parameters available to execution hooks
            such as ``resolve_execution_kwargs``; the resampling factory
            reads the ``freq`` entry from them.
    """

    injected_parameters: Mapping[str, Any] = field(default_factory=dict)

    def execute(
        self,
        context: dg.AssetExecutionContext,
        obj: T,
        ins: Mapping[str, xr.DataArray],
        **kwargs,
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Execute the indicator.

        Args:
            context: The asset execution context.
            obj: The indicator to execute.
            ins: Upstream inputs keyed by input name.
            **kwargs: Extra keyword arguments from ``resolve_execution_kwargs``.

        Returns:
            The indicator's result, with each output chunked per
            ``get_chunks``.
        """
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
        """Return the execution metadata of a materialized output value.

        Args:
            context: The asset execution context.
            value: The DataArray being materialized.

        Returns:
            The base metadata plus an ``nbytes`` entry for the output's size
            in bytes.
        """
        return {
            **super().get_metadata(context, value),
            "nbytes": dg.MetadataValue.int(value.nbytes),
        }

    def get_chunks(self, obj: xr.DataArray) -> Chunks:
        """Return the chunks to apply to an output.

        Args:
            obj: The DataArray to chunk.

        Returns:
            The chunk spec passed to ``DataArray.chunk``. Defaults to None.
        """
        return None


@dataclass(frozen=True, kw_only=True)
class XclimResamplingIndicatorFactory(XclimIndicatorFactory[xc.ResamplingIndicator]):
    """An XclimIndicatorFactory for resampling indicators.

    Injects the resampling frequency into the indicator at execution time.
    The frequency is taken from the ``freq`` entry of ``injected_parameters``
    or, when the layer uses a ``ResamplingPartitionsDefinition``, from the
    execution's partition key.
    """

    def _resolve_freq(
        self, context: dg.AssetExecutionContext, obj: xc.ResamplingIndicator
    ) -> str:
        """Resolve the resampling frequency of the indicator.

        Args:
            context: The asset execution context.
            obj: The resampling indicator.

        Returns:
            The frequency, checked against the indicator's allowed periods.
            It comes from the "freq" entry of ``injected_parameters`` if set,
            otherwise from the execution's partition key (its "freq" dimension
            for multi-partition keys).

        Raises:
            dagster.DagsterInvalidDefinitionError: If neither source is
                configured.
            ValueError: If the frequency's period is not an allowed period of
                the indicator.
        """
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
        """Resolve extra keyword arguments for ``execute``.

        Args:
            context: The asset execution context.
            obj: The resampling indicator.

        Returns:
            The "freq" keyword argument, set to the partition key of the
            resolved frequency.
        """
        return {
            XCLIM_FREQUENCY_KEYWORD: self.get_partition_key(
                context, self._resolve_freq(context, obj)
            )
        }
