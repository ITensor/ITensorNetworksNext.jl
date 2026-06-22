using Graphs: dst, src
using NamedDimsArrays: dimnames, replacedimnames
using NamedGraphs.GraphsExtensions: edge_path, post_order_dfs_edges
using TensorAlgebra: TensorAlgebra as TA

# Isometric (QR) gauge on a tree. The orthogonality center is *not* stored on the network
# — it is tracked by the caller. `orthogonalize(state, center)` canonicalizes the whole
# tree so every tensor but `center` is an isometry pointing toward `center`;
# `orthogonalize(state, source, dest)` moves the center along the tree path, returning the
# walked edges so environments can be refreshed incrementally.
#
# Each gauge step keeps the link name on its edge stable (the fresh QR bond is renamed back
# to the original link name), so name maps built against the ket — e.g. a
# `QuadraticFormNetwork`'s `link_index_map` — stay valid across gauging.

# Make `state[v]` an isometry whose only non-isometric leg points toward `w`, pushing the
# `R` factor into `state[w]`. Mutates `state`.
function gauge_move!(state, v, w)
    ln = only(linknames(state, v => w))
    tv = state[v]
    rows = collect(setdiff(dimnames(tv), [ln]))
    Q, R = TA.qr(tv, rows)
    r = only(setdiff(dimnames(Q), rows))
    new_w = R * state[w]
    setindex_preserve_graph!(state, replacedimnames(Q, r => ln), v)
    setindex_preserve_graph!(state, replacedimnames(new_w, r => ln), w)
    return state
end

"""
    orthogonalize(state, center) -> state

Canonicalize the tree tensor network `state` so that every tensor except `center` (a
vertex) is an isometry pointing toward `center`; the orthogonality center is then `center`.
"""
function orthogonalize(state, center)
    state = copy(state)
    for e in post_order_dfs_edges(state, center)
        gauge_move!(state, src(e), dst(e))
    end
    return state
end

"""
    orthogonalize(state, source, dest) -> (state, walked_edges)

Move the orthogonality center of the tree tensor network `state` from vertex `source` to
vertex `dest` by QR gauge steps along the tree path, assuming `state` is already canonical
with center `source`. Returns the re-gauged `state` and the directed edges walked (for
incremental environment refresh).
"""
function orthogonalize(state, source, dest)
    state = copy(state)
    walked = collect(edge_path(state, source, dest))
    for e in walked
        gauge_move!(state, src(e), dst(e))
    end
    return state, walked
end
