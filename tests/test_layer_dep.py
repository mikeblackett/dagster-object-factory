from dataclasses import FrozenInstanceError

import hypothesis as hp
import hypothesis.strategies as st
import pytest as pt

from dagster_object_factory.layer import Layer, LayerDep

from .strategies import layers, metadata


def test_immutability(leaf_layer: Layer) -> None:
    layer_dep = LayerDep(layer=leaf_layer)
    with pt.raises(FrozenInstanceError):
        layer_dep.layer = leaf_layer  # type: ignore[reportAttributeAccessIssue]


@hp.given(data=st.data())
def test_resolve_dependency(data: st.DataObject) -> None:
    layer = data.draw(layers())
    layer_dep = LayerDep(layer=layer)
    output_name = data.draw(st.sampled_from(sorted(layer.output_names)))
    dep_spec = layer_dep.resolve_dependency(output_name)
    assert dep_spec.output_name == output_name
    assert dep_spec.layer is layer


@hp.given(data=st.data())
def test_resolve_dependency_carries_attributes(data: st.DataObject) -> None:
    upstream = data.draw(layers())
    output_name = data.draw(st.sampled_from(sorted(upstream.output_names)))
    _metadata = data.draw(metadata())
    layer_dep = LayerDep(layer=upstream)
    spec = layer_dep.resolve_dependency(output_name, metadata=_metadata)
    assert spec.output_name == output_name
    assert spec.layer is upstream
    assert spec.key == upstream.resolve_key(output_name)
    assert spec.metadata == _metadata
    assert spec.partition_mapping is None
    # Default input_name reuses the output name.
    assert spec.resolved_input_name == output_name
