"""Components that execute xclim indicators as dagster assets."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    """An ObjectFactoryComponent that executes xclim indicators."""

    def execute(
        self,
        context: dg.AssetExecutionContext,
        obj: T,
        ins: Mapping[str, xr.DataArray],
        **kwargs: Any,
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

    def get_result_metadata(
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
            **super().get_result_metadata(context, value),
            "nbytes": dg.MetadataValue.int(value.nbytes),
        }

    def get_chunks(self, data: xr.DataArray) -> Chunks:
        """Return the chunks to apply to an output.

        Args:
            data: The DataArray to chunk.

        Returns:
            The chunk spec passed to ``DataArray.chunk``. Defaults to None.
        """
        return None

    def resolve_injected_kwargs(
        self, context: dg.ComponentLoadContext, obj: T
    ) -> dict[str, Any]:
        for name in self.injected_kwargs:
            self._validate_injectable_parameter(obj, name)
        return dict(self.injected_kwargs)

    def _validate_injectable_parameter(self, obj: xc.Indicator, name: str) -> None:
        try:
            if name in obj.injected_parameters:
                raise ValueError(f"parameter {name!r} is already injected by xclim.")
            if name not in obj.parameters:
                raise KeyError(f"{name!r} is not a parameter of {obj.identifier!r}.")
        except (ValueError, KeyError) as error:
            raise dg.DagsterInvalidDefinitionError(str(error)) from error


@dataclass(frozen=True, kw_only=True)
class XclimResamplingIndicatorFactory(XclimIndicatorFactory[xc.ResamplingIndicator]):
    """An XclimIndicatorFactory for resampling indicators.

    Resolves the resampling frequency from the ``freq`` entry of
    ``injected_kwargs`` at definition time, or, when the layer is partitioned
    by a ``ResamplingPartitionsDefinition``, from the execution's partition key.
    Supplying the frequency via both ``injected_kwargs`` and a
    ``ResamplingPartitionsDefinition`` is a definition error.
    """

    @staticmethod
    def _resampling_dimension_name(
        partitions_def: dg.PartitionsDefinition | None,
    ) -> str | None:
        """Return the name of the multi-partition dimension backed by a
        ``ResamplingPartitionsDefinition``, or ``None`` otherwise.
        """
        if isinstance(partitions_def, dg.MultiPartitionsDefinition):
            for partition_dimension in partitions_def.partitions_defs:
                if isinstance(
                    partition_dimension.partitions_def,
                    ResamplingPartitionsDefinition,
                ):
                    return partition_dimension.name
        return None

    def _is_resampling_partitioned(self) -> bool:
        """Whether the layer's partitions definition supplies the frequency."""
        return (
            isinstance(self.layer.partitions_def, ResamplingPartitionsDefinition)
            or self._resampling_dimension_name(self.layer.partitions_def) is not None
        )

    def _get_freq_partition_key(self, context: dg.AssetExecutionContext) -> str | None:
        """Return the freq string from a partition key or multi-partition key."""
        partitions_def = self.layer.partitions_def

        if isinstance(partitions_def, ResamplingPartitionsDefinition):
            return context.partition_key

        dimension = self._resampling_dimension_name(partitions_def)
        if dimension is None:
            return
        partition_key = cast(dg.MultiPartitionKey, context.partition_key)
        return partition_key.keys_by_dimension[dimension]

    def resolve_injected_kwargs(
        self, context: dg.ComponentLoadContext, obj: xc.ResamplingIndicator
    ) -> dict[str, Any]:
        kwargs = super().resolve_injected_kwargs(context, obj)
        if XCLIM_FREQUENCY_KEYWORD in kwargs and self._is_resampling_partitioned():
            raise dg.DagsterInvalidDefinitionError(
                f"{obj.identifier}: '{XCLIM_FREQUENCY_KEYWORD}' cannot be set in "
                "injected_kwargs when the layer is partitioned by a "
                "ResamplingPartitionsDefinition; supply the frequency via either "
                "the partition key or injected_kwargs, not both."
            )
        if freq := kwargs.get(XCLIM_FREQUENCY_KEYWORD):
            try:
                _validate_freq(freq, obj.allowed_periods)
            except ValueError as error:
                raise dg.DagsterInvalidDefinitionError(
                    f"{obj.identifier}: {error}"
                ) from error
        return kwargs

    def resolve_execution_kwargs(
        self, context: dg.AssetExecutionContext, obj: xc.ResamplingIndicator
    ) -> dict[str, Any]:
        kwargs = super().resolve_execution_kwargs(context, obj)
        freq = self._get_freq_partition_key(context)
        if freq:
            try:
                _validate_freq(freq, obj.allowed_periods)
            except ValueError as error:
                raise dg.DagsterInvalidDefinitionError(
                    f"{obj.identifier}: {error}"
                ) from error
            kwargs[XCLIM_FREQUENCY_KEYWORD] = freq
        return kwargs


def _validate_freq(freq: str, allowed_periods: Sequence[str] | None) -> str:
    # parse first to catch invalid freqs
    period = xcal.parse_offset(freq)[1]
    if allowed_periods is None:
        return freq
    if period not in allowed_periods:
        raise ValueError(
            f"{freq!r} has period {period!r}, "
            f"which is not allowed ({allowed_periods!r})."
        )
    return freq
