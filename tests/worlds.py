"""Hypothesis strategies for the toy world.

``worlds`` generates a coherent miniature pipeline: leaves, composites
derived from pairs of them, and collections of them, together with the
layers that declare their outputs. Every composite part and collection
member is a declared leaf, and object names are unique across kinds, so
any translator can resolve every dependency in a generated world.
"""

from dataclasses import dataclass

import hypothesis.strategies as st

from dagster_object_factory.layer import Layer, LayerDep

from .strategies import key_prefixes, names
from .toy_example import Collection, Composite, Leaf

values = st.text(min_size=1, max_size=3)


@dataclass(frozen=True)
class World:
    """A set of toy objects and the layers they translate for."""

    leaves: tuple[Leaf, ...]
    composites: tuple[Composite, ...]
    collections: tuple[Collection, ...]
    leaf_layer: Layer[str]
    composite_layer: Layer[str]
    collection_layer: Layer[str]


@st.composite
def _leaves(draw: st.DrawFn, min_size: int = 1, max_size: int = 4) -> tuple[Leaf, ...]:
    leaf_names = draw(
        st.lists(names(), min_size=min_size, max_size=max_size, unique=True)
    )
    return tuple(Leaf(name=name, value=draw(values)) for name in leaf_names)


@st.composite
def worlds(draw: st.DrawFn) -> World:
    """A world of toy objects, as described in the module docstring."""
    leaves = draw(_leaves(min_size=2, max_size=4))
    taken = {leaf.name for leaf in leaves}

    composite_names = draw(
        st.lists(
            names().filter(lambda candidate: candidate not in taken),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    composites = []
    for name in composite_names:
        parts = draw(
            st.lists(st.sampled_from(leaves), min_size=2, max_size=2, unique=True)
        )
        composites.append(Composite(name=name, parts=(parts[0], parts[1])))
    taken.update(composite.name for composite in composites)

    # `taken` already includes the composite names, so the filter closes
    # over the right set by the time this draw happens.
    collection_names = draw(
        st.lists(
            names().filter(lambda candidate: candidate not in taken),
            min_size=0,
            max_size=2,
            unique=True,
        )
    )
    collections = []
    for name in collection_names:
        members = draw(
            st.lists(st.sampled_from(leaves), min_size=1, max_size=3, unique=True)
        )
        collections.append(Collection(name=name, members=tuple(members)))

    leaf_layer = Layer(
        name="leaf",
        key_prefix=tuple(draw(key_prefixes())),
        python_type=str,
        output_names=frozenset(leaf.name for leaf in leaves),
    )
    composite_layer = Layer(
        name="composite",
        key_prefix=tuple(draw(key_prefixes())),
        python_type=str,
        output_names=frozenset(composite.name for composite in composites),
        layer_deps=(LayerDep(layer=leaf_layer),),
    )
    collection_layer = Layer(
        name="collection",
        key_prefix=tuple(draw(key_prefixes())),
        python_type=str,
        output_names=frozenset(
            member.name for collection in collections for member in collection.members
        ),
        layer_deps=(LayerDep(layer=leaf_layer),),
    )
    return World(
        leaves=leaves,
        composites=tuple(composites),
        collections=tuple(collections),
        leaf_layer=leaf_layer,
        composite_layer=composite_layer,
        collection_layer=collection_layer,
    )
