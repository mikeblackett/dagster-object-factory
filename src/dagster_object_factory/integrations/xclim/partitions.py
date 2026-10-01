import dagster as dg


class ResamplingPartitionsDefinition(dg.StaticPartitionsDefinition):
    """A static partitions definition whose partitions are resampling frequencies."""
