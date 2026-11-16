from dataclasses import FrozenInstanceError
from types import EllipsisType

import dagster as dg
import hypothesis as hp
import hypothesis.strategies as st
import pytest as pt

from dagster_object_factory.layer import DependencySpec, Layer, LayerDep

from .strategies import dependency_specs, layers, metadata
from .toy_example import B


def test_immutability(leaf_layer: Layer) -> None:
    dep_spec = DependencySpec(layer=leaf_layer, output_name=B.name)
    with pt.raises(FrozenInstanceError):
        dep_spec.output_name = "other"  # type: ignore[reportAttributeAccessIssue]


@hp.given(data=st.data())
def test_resolve_dependency(data: st.DataObject) -> None:
    layer = data.draw(layers())
    layer_dep = LayerDep(layer=layer)
    output_name = data.draw(st.sampled_from(sorted(layer.output_names)))

    dep_spec = layer_dep.resolve_dependency(output_name)
    assert dep_spec.output_name == output_name
    assert dep_spec.layer is layer


@hp.given(data=st.data())
def test_key_resolves_from_layer(data: st.DataObject) -> None:
    layer = data.draw(layers())
    dep_spec = data.draw(dependency_specs(layer))
    assert dep_spec.layer is layer
    assert dep_spec.output_name in layer
    assert dep_spec.key == layer.resolve_key(dep_spec.output_name)


def test_undeclared_output_raises_value_error(leaf_layer: Layer) -> None:
    with pt.raises(ValueError, match="not a declared output"):
        DependencySpec(output_name="not-there", layer=leaf_layer)


@pt.mark.parametrize(
    "input_name, resolved_input_name",
    [
        # fallback to output name
        (..., "output"),
        # No input name
        (None, None),
        # Explicit input name
        ("explicit", "explicit"),
    ],
)
def test_resolved_input_name_computed(
    input_name: str | EllipsisType | None, resolved_input_name: str | None
) -> None:
    layer = Layer(
        name="layer",
        python_type=str,
        key_prefix=(),
        output_names=frozenset({"output"}),
    )
    dep_spec = DependencySpec(output_name="output", layer=layer, input_name=input_name)
    assert dep_spec.resolved_input_name == resolved_input_name


@hp.given(data=st.data())
def test_to_asset_dep(data: st.DataObject) -> None:
    layer = data.draw(layers())
    dep_spec = data.draw(dependency_specs(layer))
    key = layer.resolve_key(dep_spec.output_name)
    asset_dep = dep_spec.to_dep()
    assert isinstance(asset_dep, dg.AssetDep)
    assert asset_dep.asset_key == key


@hp.given(data=st.data())
def test_to_dep_merges_metadata(data: st.DataObject) -> None:
    layer = data.draw(layers())
    output_name = data.draw(st.sampled_from(sorted(layer.output_names)))
    call_metadata = data.draw(metadata())
    spec_metadata = data.draw(metadata())
    dep_spec = DependencySpec(
        output_name=output_name, layer=layer, metadata=spec_metadata
    )
    expected = {**(spec_metadata or {}), **(call_metadata or {})}
    assert dep_spec.to_dep(metadata=call_metadata).metadata == expected


@hp.given(data=st.data())
def test_to_asset_in(data: st.DataObject) -> None:
    layer = data.draw(layers())
    dep_spec = data.draw(dependency_specs(layer))
    key = layer.resolve_key(dep_spec.output_name)
    asset_in = dep_spec.to_in()
    assert isinstance(asset_in, dg.AssetIn)
    assert asset_in.key == key


@hp.given(data=st.data())
def test_to_asset_in_dagster_type(data: st.DataObject) -> None:
    layer = data.draw(layers())
    dep_spec = data.draw(dependency_specs(layer))
    asset_in = dep_spec.to_in()
    output_value = data.draw(st.from_type(layer.python_type))
    assert isinstance(asset_in, dg.AssetIn)
    assert isinstance(asset_in.dagster_type, dg.DagsterType)
    assert dg.check_dagster_type(asset_in.dagster_type, output_value)


@hp.given(data=st.data())
def test_to_in_merges_metadata(data: st.DataObject) -> None:
    layer = data.draw(layers())
    output_name = data.draw(st.sampled_from(sorted(layer.output_names)))
    call_metadata = data.draw(metadata())
    spec_metadata = data.draw(metadata())
    dep_spec = DependencySpec(
        output_name=output_name, layer=layer, metadata=spec_metadata
    )
    expected = {**(spec_metadata or {}), **(call_metadata or {})}
    assert dep_spec.to_in(metadata=call_metadata).metadata == expected


def test_partition_mapping_from_call(leaf_layer: Layer) -> None:
    mapping = dg.IdentityPartitionMapping()
    dep_spec = DependencySpec(
        output_name=B.name,
        layer=leaf_layer,
        partition_mapping=mapping,
    )
    assert dep_spec.to_dep().partition_mapping == mapping
    assert dep_spec.to_in().partition_mapping == mapping


def test_partition_mapping_from_layer(leaf_layer: Layer) -> None:
    mapping = dg.IdentityPartitionMapping()
    layer_dep = LayerDep(layer=leaf_layer, partition_mapping=mapping)
    spec = layer_dep.resolve_dependency(B.name)
    assert spec.partition_mapping == mapping
