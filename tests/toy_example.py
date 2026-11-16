"""A small abstract world of named objects for the tests.

A ``Leaf`` carries an intrinsic value. A ``Composite`` is derived from
exactly two leaves, and its value is a deterministic join of theirs. A
``Collection`` groups leaves, one output per member. The concrete instances
below stand in for a real project's objects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Leaf:
    """A named object with an intrinsic value."""

    name: str
    value: str


@dataclass(frozen=True)
class Composite:
    """An object derived from exactly two leaves.

    Its value is a deterministic join of its parts' values.
    """

    name: str
    parts: tuple[Leaf, Leaf]

    def value(self) -> str:
        return "+".join(sorted(part.value for part in self.parts))


@dataclass(frozen=True)
class Collection:
    """A named group of leaves, one output per member."""

    name: str
    members: tuple[Leaf, ...]

    def value(self) -> tuple[str, ...]:
        return tuple(member.value for member in self.members)


A = Leaf(name="a", value="1")
B = Leaf(name="b", value="2")
C = Leaf(name="c", value="3")
AB = Composite(name="ab", parts=(A, B))
BC = Composite(name="bc", parts=(B, C))
CA = Composite(name="ca", parts=(C, A))
GROUP = Collection(name="group", members=(A, B, C))
