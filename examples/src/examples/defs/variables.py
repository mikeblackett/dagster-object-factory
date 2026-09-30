import dagster as dg
import xarray as xr

from dagster_object_factory import Layer

_resampler = xr.tutorial.open_dataset("air_temperature", cache=True).air.resample(
    time="D"
)

layer = Layer(
    name="variables",
    key_prefix=("variables",),
    output_names=frozenset({"tas", "tasmin", "tasmax", "ua"}),
    python_type=xr.DataArray,
)


@dg.asset(key_prefix=layer.key_prefix)
def tas() -> xr.DataArray:
    return _resampler.mean()


@dg.asset(key_prefix=layer.key_prefix)
def tasmin() -> xr.DataArray:
    return _resampler.min()


@dg.asset(key_prefix=layer.key_prefix)
def tasmax() -> xr.DataArray:
    return _resampler.max()


@dg.asset(key_prefix=layer.key_prefix)
def ua() -> xr.DataArray:
    return xr.DataArray()
