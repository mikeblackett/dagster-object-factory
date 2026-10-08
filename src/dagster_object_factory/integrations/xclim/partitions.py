from collections.abc import Sequence

import dagster as dg
import pandas as pd


class ResamplingPartitionsDefinition(dg.StaticPartitionsDefinition):
    """A static partitions definition whose partitions are resampling frequencies."""

    def __init__(self, partition_keys: Sequence[str]):
        try:
            for key in partition_keys:
                pd.tseries.frequencies.to_offset(key)
        except ValueError as error:
            raise dg.DagsterInvalidDefinitionError(str(error)) from error

        super().__init__(partition_keys)
