def patch_torch_pytree_compat():
    """Bridge newer transformers imports on torch versions before 2.2."""
    try:
        import torch
    except ImportError:
        return

    pytree = getattr(getattr(torch, "utils", None), "_pytree", None)
    if pytree is None or hasattr(pytree, "register_pytree_node"):
        return
    if not hasattr(pytree, "_register_pytree_node"):
        return

    def register_pytree_node(node_type, flatten_fn, unflatten_fn, **kwargs):
        kwargs.pop("serialized_type_name", None)
        kwargs.pop("to_dumpable_context", None)
        kwargs.pop("from_dumpable_context", None)
        return pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn, **kwargs)

    pytree.register_pytree_node = register_pytree_node


patch_torch_pytree_compat()
