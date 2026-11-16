"""Tests for ObjectFactoryComponent."""

import string
from collections.abc import Iterable, Mapping
from typing import Any, cast

import dagster as dg
import hypothesis as hp
import hypothesis.strategies as st
import pytest as pt
from dagster._core.events import StepMaterializationData

from dagster_object_factory import DependencySpec, Layer

from .conftest import (
    AB,
    BC,
    CA,
    GROUP,
    LEAVES,
    A,
    B,
)
from .toy_example import Composite, Leaf
from .toy_integration import (
    CollectionComponent,
    CollectionTranslator,
    CompositeComponent,
    CompositeTranslator,
    LeafComponent,
    LeafTranslator,
)
from .worlds import worlds

OWNERS = st.lists(
    st.sampled_from(["team:data", "team:infra", "mike@example.com"]), unique=True
)

# Dagster validates tag and metadata keys against [A-Za-z0-9_.-]{1,63}, and
# tag values against [A-Za-z0-9_.-]{0,63}, so the generated keys and values
# are restricted to that alphabet (the namespace slash is excluded, which
# would risk the reserved system prefix).
_ATTR_ALPHABET = string.ascii_lowercase + string.digits + "_-."
ATTR_KEYS = st.text(alphabet=_ATTR_ALPHABET, min_size=1, max_size=3)
ATTR_VALUES = st.text(alphabet=_ATTR_ALPHABET, min_size=0, max_size=5)


def _leaf_component(leaf_layer: Layer, **kwargs: Any) -> LeafComponent:
    return LeafComponent(
        translator=LeafTranslator(leaf_layer), objects=list(LEAVES), **kwargs
    )


def _load_context() -> dg.ComponentLoadContext:
    # Hypothesis forbids function-scoped fixtures in @given tests, so the
    # property tests build a load context inline.
    return dg.ComponentTree.for_test().load_context


def _assets(defs: dg.Definitions) -> list[dg.AssetsDefinition]:
    """Narrow ``Definitions.assets`` to ``AssetsDefinition`` values.

    Dagster types ``assets`` as an optional heterogeneous iterable, but
    ``build_defs`` only produces ``AssetsDefinition`` objects.
    """
    assets = defs.assets
    assert assets is not None
    assert all(isinstance(asset, dg.AssetsDefinition) for asset in assets)
    return cast("list[dg.AssetsDefinition]", list(assets))


## Definition-level behavior


def test_layer_convenience(leaf_layer: Layer) -> None:
    component = _leaf_component(leaf_layer)
    assert component.layer is leaf_layer


def test_build_defs_returns_definitions(
    load_context: dg.ComponentLoadContext, leaf_layer: Layer
) -> None:
    defs = _leaf_component(leaf_layer).build_defs(load_context)
    assert isinstance(defs, dg.Definitions)
    assets = _assets(defs)
    assert len(assets) == len(LEAVES)
    assert {key.to_user_string() for asset in assets for key in asset.keys} == {
        "item/a",
        "item/b",
        "item/c",
    }


def test_make_asset_identity(
    load_context: dg.ComponentLoadContext, leaf_layer: Layer
) -> None:
    asset = _leaf_component(leaf_layer).make_asset(load_context, A)
    assert asset.keys == {dg.AssetKey(["item", "a"])}
    assert asset.node_def.name == "a"
    assert asset.node_def.description is None
    assert asset.group_names_by_key == {dg.AssetKey(["item", "a"]): "leaf"}


def test_make_asset_ins_keyed_by_resolved_input_name(
    load_context: dg.ComponentLoadContext, composite_layer: Layer
) -> None:
    class _RenamedInputs(CompositeTranslator):
        def resolve_dependency_specs(
            self, obj: Composite, /
        ) -> Iterable[DependencySpec]:
            for part in obj.parts:
                layer_dep = self.layer.resolve_layer_dep(part.name)
                assert layer_dep is not None
                yield layer_dep.resolve_dependency(
                    part.name, input_name=f"{part.name}_in"
                )

    component = CompositeComponent(
        translator=_RenamedInputs(composite_layer), objects=[BC]
    )
    asset = component.make_asset(load_context, BC)
    assert set(asset.input_names) == {"b_in", "c_in"}


def test_make_asset_inputless_specs_become_deps(
    load_context: dg.ComponentLoadContext, composite_layer: Layer
) -> None:
    class _Inputless(CompositeTranslator):
        def resolve_dependency_specs(
            self, obj: Composite, /
        ) -> Iterable[DependencySpec]:
            for index, part in enumerate(obj.parts):
                layer_dep = self.layer.resolve_layer_dep(part.name)
                assert layer_dep is not None
                # The first part is input-less; the rest feed inputs.
                input_name = None if index == 0 else part.name
                yield layer_dep.resolve_dependency(part.name, input_name=input_name)

    component = CompositeComponent(translator=_Inputless(composite_layer), objects=[CA])
    asset = component.make_asset(load_context, CA)
    # CA is c + a: c is a plain dep, a feeds a named input.
    # Dagster surfaces input-less deps as inputs under a name generated from
    # the asset key (e.g. "item_c"), so assert on the library contract:
    # the dep's raw output name must not appear as a named input.
    assert "a" in asset.input_names
    assert "c" not in asset.input_names
    assert asset.dependency_keys == {
        dg.AssetKey(["item", "c"]),
        dg.AssetKey(["item", "a"]),
    }


def test_build_defs_merges_component_attributes(
    load_context: dg.ComponentLoadContext, leaf_layer: Layer
) -> None:
    component = _leaf_component(
        leaf_layer,
        deps=[dg.AssetDep(asset=dg.AssetKey(["upstream", "dep"]))],
        metadata={"component": "meta"},
        owners=["team:data"],
        tags={"component": "tag"},
        kinds={"sql"},
    )
    defs = component.build_defs(load_context)
    for asset in _assets(defs):
        key = next(iter(asset.keys))
        assert asset.metadata_by_key[key] == {"component": "meta"}
        assert asset.owners_by_key[key] == ["team:data"]
        assert asset.tags_by_key[key]["component"] == "tag"
        assert asset.specs_by_key[key].kinds == {"sql"}
        assert dg.AssetKey(["upstream", "dep"]) in asset.dependency_keys


def test_build_defs_merge_overrides_translation_attributes(
    load_context: dg.ComponentLoadContext, leaf_layer: Layer
) -> None:
    class _TaggingTranslator(LeafTranslator):
        def get_tags(self, obj: Leaf, /) -> Mapping[str, str]:
            return {"shared": "translation", "only": "translation"}

        def get_metadata(self, obj: Leaf, /) -> Mapping[str, Any]:
            return {"shared": "translation", "only": "translation"}

    component = LeafComponent(
        translator=_TaggingTranslator(leaf_layer),
        objects=[A],
        tags={"shared": "component"},
        metadata={"shared": "component"},
    )
    (asset,) = _assets(component.build_defs(load_context))
    key = next(iter(asset.keys))
    # Component values win on conflict; translation-only keys survive.
    assert asset.tags_by_key[key] == {
        "shared": "component",
        "only": "translation",
    }
    assert asset.metadata_by_key[key] == {
        "shared": "component",
        "only": "translation",
    }


def test_make_asset_applies_io_manager_key(
    load_context: dg.ComponentLoadContext, leaf_layer: Layer
) -> None:
    component = _leaf_component(leaf_layer, io_manager_key="custom")
    asset = component.make_asset(load_context, A)
    assert (
        asset.get_io_manager_key_for_asset_key(dg.AssetKey(["item", "a"])) == "custom"
    )


def test_make_asset_carries_layer_partitions_def(
    load_context: dg.ComponentLoadContext, leaf_layer: Layer
) -> None:
    partitions_def = dg.TimeWindowPartitionsDefinition(
        start="2024-01-01",
        fmt="%Y-%m-%d",
        end="2024-01-05",
        cron_schedule="0 0 * * *",
        timezone="UTC",
    )
    layer = Layer(
        name="leaf",
        key_prefix=("item",),
        python_type=str,
        output_names=frozenset(leaf.name for leaf in LEAVES),
        partitions_def=partitions_def,
    )
    asset = _leaf_component(layer).make_asset(load_context, A)
    assert asset.partitions_def is partitions_def


def test_resolve_injected_kwargs_returns_copy(
    load_context: dg.ComponentLoadContext, leaf_layer: Layer
) -> None:
    component = _leaf_component(leaf_layer, injected_kwargs={"injected": "value"})
    resolved = component.resolve_injected_kwargs(load_context, A)
    assert resolved == {"injected": "value"}
    assert resolved is not component.injected_kwargs


def test_resolve_injected_kwargs_fail_fast(
    load_context: dg.ComponentLoadContext, leaf_layer: Layer
) -> None:
    class _Validating(LeafComponent):
        def resolve_injected_kwargs(
            self, context: dg.ComponentLoadContext, obj: Leaf
        ) -> dict[str, Any]:
            if obj.name == A.name:
                raise ValueError("a is not allowed")
            return super().resolve_injected_kwargs(context, obj)

    component = _Validating(translator=LeafTranslator(leaf_layer), objects=[A])
    with pt.raises(ValueError, match="a is not allowed"):
        component.build_defs(load_context)


def test_required_resource_keys(
    load_context: dg.ComponentLoadContext, leaf_layer: Layer
) -> None:
    class _Resourced(LeafComponent):
        def get_required_resource_keys(self) -> frozenset[str]:
            return frozenset({"database"})

    component = _Resourced(translator=LeafTranslator(leaf_layer), objects=[A])
    asset = component.make_asset(load_context, A)
    assert "database" in asset.required_resource_keys


# --- Property-based definition-level behavior -----------------------------


@hp.given(world=worlds())
def test_build_defs_world(world) -> None:
    load_context = _load_context()
    component = CompositeComponent(
        translator=CompositeTranslator(world.composite_layer),
        objects=list(world.composites),
    )
    defs = component.build_defs(load_context)
    assets = _assets(defs)
    expected = {
        world.composite_layer.resolve_key(composite.name)
        for composite in world.composites
    }
    assert len(assets) == len(world.composites)
    assert {key for asset in assets for key in asset.keys} == expected


@hp.given(
    world=worlds(),
    tags=st.dictionaries(ATTR_KEYS, ATTR_VALUES),
    metadata=st.dictionaries(ATTR_KEYS, st.text()),
    owners=OWNERS,
    kinds=st.frozensets(st.sampled_from(["sql", "python", "api"])),
)
def test_build_defs_world_merges_attributes(
    world,
    tags: dict[str, str],
    metadata: dict[str, str],
    owners: list[str],
    kinds: frozenset[str],
) -> None:
    load_context = _load_context()
    component = CompositeComponent(
        translator=CompositeTranslator(world.composite_layer),
        objects=list(world.composites),
        tags=tags,
        metadata=metadata,
        owners=owners,
        kinds=set(kinds),
    )
    defs = component.build_defs(load_context)
    for asset in _assets(defs):
        for key in asset.keys:
            # The dagster/kind tags are derived from kinds, so the asset's
            # tags are a superset of the component's.
            assert tags.items() <= asset.tags_by_key[key].items()
            assert asset.metadata_by_key[key] == metadata
            assert asset.owners_by_key[key] == owners
            assert asset.specs_by_key[key].kinds == kinds


# --- End-to-end execution --------------------------------------------------


def test_materialize_end_to_end(
    instance,
    load_context: dg.ComponentLoadContext,
    leaf_layer: Layer,
    composite_layer: Layer,
) -> None:
    executed_ins: dict[str, dict[str, Any]] = {}

    class _SpyComposite(CompositeComponent):
        def execute(
            self,
            context: dg.AssetExecutionContext,
            obj: Composite,
            ins: Mapping[str, Any],
            **kwargs: Any,
        ) -> str:
            executed_ins[obj.name] = dict(ins)
            return obj.value()

    leaf_component = _leaf_component(leaf_layer)
    composite_component = _SpyComposite(
        translator=CompositeTranslator(composite_layer), objects=[AB]
    )
    assets = [leaf_component.make_asset(load_context, leaf) for leaf in LEAVES]
    assets.append(composite_component.make_asset(load_context, AB))
    result = dg.materialize(assets, instance=instance)
    assert result.success
    assert result.asset_value(dg.AssetKey(["item", "ab"])) == AB.value()
    # The upstream values flowed into execute under the resolved input names.
    assert executed_ins["ab"] == {"a": A.value, "b": B.value}


def test_materialize_collection_multi_output(
    instance,
    load_context: dg.ComponentLoadContext,
    leaf_layer: Layer,
    collection_layer: Layer,
) -> None:
    leaf_component = _leaf_component(leaf_layer)
    collection_component = CollectionComponent(
        translator=CollectionTranslator(collection_layer), objects=[GROUP]
    )
    collection_asset = collection_component.make_asset(load_context, GROUP)
    assert collection_asset.keys == {
        dg.AssetKey(["group", "a"]),
        dg.AssetKey(["group", "b"]),
        dg.AssetKey(["group", "c"]),
    }
    assets = [leaf_component.make_asset(load_context, leaf) for leaf in LEAVES]
    assets.append(collection_asset)
    result = dg.materialize(assets, instance=instance)
    assert result.success
    # One materialized value per output: the value of each member leaf.
    assert (
        result.asset_value(dg.AssetKey(["group", "a"])) == A.value == GROUP.value()[0]
    )
    assert (
        result.asset_value(dg.AssetKey(["group", "b"])) == B.value == GROUP.value()[1]
    )


def test_materialize_partitioned(
    instance, load_context: dg.ComponentLoadContext
) -> None:
    partitions_def = dg.TimeWindowPartitionsDefinition(
        start="2024-01-01",
        fmt="%Y-%m-%d",
        end="2024-01-05",
        cron_schedule="0 0 * * *",
        timezone="UTC",
    )
    layer = Layer(
        name="leaf",
        key_prefix=("item",),
        python_type=str,
        output_names=frozenset(leaf.name for leaf in LEAVES),
        partitions_def=partitions_def,
    )
    component = _leaf_component(layer)
    assets = [component.make_asset(load_context, leaf) for leaf in LEAVES]
    result = dg.materialize(assets, instance=instance, partition_key="2024-01-02")
    assert result.success
    assert result.asset_value(dg.AssetKey(["item", "a"])) == A.value


def test_execute_receives_injected_and_execution_kwargs(
    instance, load_context: dg.ComponentLoadContext, leaf_layer: Layer
) -> None:
    class _KwargEcho(LeafComponent):
        def resolve_execution_kwargs(
            self, context: dg.AssetExecutionContext, obj: Leaf
        ) -> dict[str, Any]:
            return {"shared": "execution"}

        def execute(
            self,
            context: dg.AssetExecutionContext,
            obj: Leaf,
            ins: Mapping[str, Any],
            **kwargs: Any,
        ) -> str:
            return str(sorted(kwargs.items()))

    component = _KwargEcho(
        translator=LeafTranslator(leaf_layer),
        objects=[A],
        injected_kwargs={"shared": "injected", "static": "yes"},
    )
    asset = component.make_asset(load_context, A)
    result = dg.materialize([asset], instance=instance)
    assert result.success
    # Execution kwargs merge over injected kwargs, which merge over nothing.
    assert result.asset_value(dg.AssetKey(["item", "a"])) == str(
        sorted([("shared", "execution"), ("static", "yes")])
    )


def test_result_metadata_attached(
    instance, load_context: dg.ComponentLoadContext, leaf_layer: Layer
) -> None:
    class _Metadata(LeafComponent):
        def get_result_metadata(
            self, context: dg.AssetExecutionContext, value: str
        ) -> dict[str, Any]:
            return {"materialized": value}

    component = _Metadata(translator=LeafTranslator(leaf_layer), objects=[A])
    asset = component.make_asset(load_context, A)
    result = dg.materialize([asset], instance=instance)
    assert result.success
    events = {
        event.asset_key: event for event in result.get_asset_materialization_events()
    }
    data = events[dg.AssetKey(["item", "a"])].event_specific_data
    assert isinstance(data, StepMaterializationData)
    metadata = data.materialization.metadata
    assert metadata["materialized"].value == A.value
