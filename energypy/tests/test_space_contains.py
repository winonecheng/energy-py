

from energypy.common.spaces.composite import ActionSpace


from energypy.common.spaces import PrimitiveConfig as Prim

discr = ActionSpace('action').from_primitives(
    Prim('a', 0, 1, type='continuous', data=None)
)
