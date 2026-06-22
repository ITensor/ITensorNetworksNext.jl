import .AlgorithmsInterfaceExtensions as AIE
import AlgorithmsInterface as AI
using .LazyNamedDimsArrays: lazy
using Base: @kwdef
using DataGraphs: DataGraphs, underlying_graph
using Graphs: Graphs, dst, edges, edgetype, neighbors, src, vertices
using KrylovKit: eigsolve
using NamedDimsArrays:
    NamedDimsArrays as NDA, AbstractNamedDimsArray, dimnames, replacedimnames
using NamedGraphs.GraphsExtensions: edge_path, post_order_dfs_edges, vertextype
using NamedGraphs: NamedGraphs
using TensorAlgebra: TensorAlgebra as TA
using VectorInterface: VectorInterface as VI

# ============================ TensorNetworkOperator =====================================

struct TensorNetworkOperator{V, VD, State, SiteMap} <: AbstractTensorNetwork{V, VD}
    state::State
    site_index_map::SiteMap
end

# `state` is the underlying operator `TensorNetwork`, mirroring `NamedDimsArrays.state` on a
# `NamedDimsOperator`. The rest forwards the tensor-network interface to it.
NDA.state(o::TensorNetworkOperator) = o.state
site_index_map(o::TensorNetworkOperator) = o.site_index_map

DataGraphs.underlying_graph(o::TensorNetworkOperator) = underlying_graph(NDA.state(o))
function DataGraphs.is_vertex_assigned(o::TensorNetworkOperator, v)
    return DataGraphs.is_vertex_assigned(NDA.state(o), v)
end
function DataGraphs.is_edge_assigned(o::TensorNetworkOperator, e)
    return DataGraphs.is_edge_assigned(NDA.state(o), e)
end
function DataGraphs.get_vertex_data(o::TensorNetworkOperator, v)
    return DataGraphs.get_vertex_data(NDA.state(o), v)
end

function Base.copy(o::TensorNetworkOperator)
    return TensorNetworkOperator(copy(NDA.state(o)), site_index_map(o))
end

function TensorNetworkOperator(state, site_index_map)
    return TensorNetworkOperator{
        vertextype(state), eltype(state), typeof(state), typeof(site_index_map),
    }(
        state, site_index_map
    )
end

# ============================ QuadraticFormNetwork ======================================

struct QuadraticFormNetwork{V, VD, Ket, Operator, LinkMap} <:
    AbstractTensorNetwork{V, VD}
    ket::Ket
    operator::Operator
    link_index_map::LinkMap
end

function bra_name_map(qf::QuadraticFormNetwork)
    return merge(site_index_map(qf.operator), qf.link_index_map)
end

ket_tensor(qf::QuadraticFormNetwork, v) = qf.ket[v]
operator_tensor(qf::QuadraticFormNetwork, v) = qf.operator[v]
function bra_tensor(qf::QuadraticFormNetwork, v)
    m = bra_name_map(qf)
    return replacedimnames(n -> get(m, n, n), conj(qf.ket[v]))
end

# === AbstractTensorNetwork / DataGraphs interface ===

DataGraphs.underlying_graph(qf::QuadraticFormNetwork) = underlying_graph(qf.ket)

function DataGraphs.is_vertex_assigned(qf::QuadraticFormNetwork, v)
    return DataGraphs.is_vertex_assigned(qf.ket, v)
end
DataGraphs.is_edge_assigned(::QuadraticFormNetwork, _e) = false

function DataGraphs.get_vertex_data(qf::QuadraticFormNetwork, v)
    return lazy(ket_tensor(qf, v)) * lazy(operator_tensor(qf, v)) * lazy(bra_tensor(qf, v))
end

function Base.copy(qf::QuadraticFormNetwork)
    return QuadraticFormNetwork(copy(qf.ket), copy(qf.operator), qf.link_index_map)
end

# === constructor ===

function QuadraticFormNetwork(ket, operator::TensorNetworkOperator, link_index_map)
    V = vertextype(ket)
    tmp = QuadraticFormNetwork{
        V, Any, typeof(ket), typeof(operator), typeof(link_index_map),
    }(
        ket,
        operator,
        link_index_map
    )
    VD = typeof(DataGraphs.get_vertex_data(tmp, first(vertices(ket))))
    return QuadraticFormNetwork{
        V, VD, typeof(ket), typeof(operator), typeof(link_index_map),
    }(
        ket,
        operator,
        link_index_map
    )
end

# ============================ Environments ==============================================
#
# On a tree, the projected-Hamiltonian environment of `⟨ψ|H|ψ⟩` is exact: the message on
# directed edge `v → w` is the contraction of the entire `⟨ψ|H|ψ⟩` subtree on `v`'s side
# of the cut `(v, w)`. It carries the three bonds crossing the cut (ket, operator, bra).
#
# The messages are computed by a single dependency-ordered pass over every directed edge
# (`forest_cover_edge_sequence` visits each tree edge in both directions, sources before
# targets), with each message an exact contraction of `qf[v]` against the already-computed
# incoming messages from `v`'s other neighbors. No fixed-point iteration is needed.

function incoming_subtree_messages(messages, graph, v, w)
    return [
        NDA.state(messages[edgetype(graph)(u, v)]) for
            u in neighbors(graph, v) if u != w
    ]
end

function environment_operator(message, link_index_map)
    ketnames = [n for n in dimnames(message) if haskey(link_index_map, n)]
    branames = [link_index_map[n] for n in ketnames]
    return NDA.operator(message, branames, ketnames)
end

"""
    quadratic_form_environments(qf::QuadraticFormNetwork; root) -> MessageCache

Exact projected-Hamiltonian environments of `⟨ψ|H|ψ⟩` on a tree, as a `MessageCache` of
`NamedDimsArrays` operators keyed by directed edges. The message on `v → w` is the
contraction of the `⟨ψ|H|ψ⟩` subtree on `v`'s side of `(v, w)`, wrapped as an operator
recording the bra ↔ ket link correspondence (see [`environment_operator`](@ref)).
"""
function quadratic_form_environments(qf::QuadraticFormNetwork; kwargs...)
    sequence = forest_cover_edge_sequence(qf; kwargs...)
    messages = Dict{edgetype(qf), Any}()
    for e in sequence
        v, w = src(e), dst(e)
        incoming = incoming_subtree_messages(messages, qf, v, w)
        message = contract_network([[qf[v]]; incoming])
        messages[e] = environment_operator(message, qf.link_index_map)
    end
    return messagecache(messages)
end

# ============================ Effective Hamiltonian =====================================
#
# The effective Hamiltonian for a `region` (a vector of vertices) is the projected operator
# obtained by contracting the operator tensors on the region with the incoming environment
# messages on the region's boundary. Contracting those gives a single tensor whose codomain
# (output) names are the region's bra names and whose domain (input) names are its ket names
# — i.e. exactly a `NamedDimsArrays` operator. So `effective_hamiltonian` returns that
# operator, and its action on a region ket tensor `T` is `NamedDimsArrays.apply(H, T)` (which
# contracts the ket names and renames the resulting bra names back to ket names). Returning
# an operator object (rather than a bare closure) follows the effective-Hamiltonian designs
# in ITensorMPS (`ProjMPO`), MPSKit (`AC_hamiltonian`), and TeNPy (`EffectiveH`).

"""
    effective_hamiltonian(qf::QuadraticFormNetwork, env, region) -> NamedDimsOperator

Effective (projected) Hamiltonian for `region` (a vector of vertices) as a `NamedDimsArrays`
operator: its domain (input) names are the region's ket names and its codomain (output)
names are the matching bra names. Apply it to a region ket tensor `T` with
`NamedDimsArrays.apply`. The environment `env` is a `MessageCache` of
[`quadratic_form_environments`](@ref).
"""
function effective_hamiltonian(qf::QuadraticFormNetwork, env, region)
    operators = [operator_tensor(qf, v) for v in region]
    boundary = [NDA.state(m) for m in incoming_edge_data(env, region)]
    h = contract_network([operators; boundary])
    sitemap = site_index_map(qf.operator)
    ketnames = [
        n for n in dimnames(h) if haskey(sitemap, n) || haskey(qf.link_index_map, n)
    ]
    branames = [bra_name_map(qf)[n] for n in ketnames]
    return NDA.operator(h, branames, ketnames)
end

# ============================ Orthogonalization =========================================
#
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

# ============================ Local update (TwoSiteEigsolve) ============================
#
# Per-region update strategy, resolved through `select_algorithm`/`default_algorithm` like
# the belief-propagation and apply-operator strategies. The default `TwoSiteEigsolve`
# extracts the region's two-site ket tensor, finds the lowest eigenpair of the effective
# Hamiltonian with `KrylovKit.eigsolve`, and SVD-splits the result back onto the two
# vertices (truncating with `trunc`). Assumes the orthogonality center sits on `region`, so
# the effective Hamiltonian is the exact projected Hamiltonian and the eigenvalue is the
# energy. Mutates the ket of `qf`.

abstract type DMRGUpdateAlgorithm <: AbstractAlgorithm end

function region_update! end

@kwdef struct TwoSiteEigsolve{Trunc} <: DMRGUpdateAlgorithm
    trunc::Trunc = nothing
    which::Symbol = :SR
end

function default_algorithm(::typeof(region_update!), ::Type{<:Tuple}; kwargs...)
    return TwoSiteEigsolve(; kwargs...)
end

"""
    region_update!(qf::QuadraticFormNetwork, env, region; alg=nothing, kwargs...) -> (qf, energy)

Optimize the ket tensors on `region` (a vector of vertices) against the effective
Hamiltonian of `qf` with environments `env`, returning the updated `qf` and the energy.
The orthogonality center must sit on `region`.
"""
function region_update!(qf, env, region; alg = nothing, kwargs...)
    algorithm = select_algorithm(region_update!, alg, (qf, env, region); kwargs...)
    return region_update!(algorithm, qf, env, region)
end

function region_update!(alg::TwoSiteEigsolve, qf, env, region)
    return region_update_nsite!(Val(length(region)), alg, qf, env, region)
end

function region_update_nsite!(::Val{N}, alg, qf, env, region) where {N}
    return throw(ArgumentError("$N-site region update not implemented"))
end

function svd_split(T, rows; trunc)
    return isnothing(trunc) ? TA.svd(T, rows) : TA.svd(T, rows; trunc)
end

# --- Temporary `VectorInterface` overloads for named arrays ---------------------------
#
# `KrylovKit.eigsolve` drives its Krylov vectors through `VectorInterface`. The generic
# `AbstractArray` fallbacks broadcast in a way that fails on a named array (e.g.
# `zerovector(x, S)`), so we provide name-aware methods here. This is type piracy on
# `AbstractNamedDimsArray` and is intended to move into `NamedDimsArrays` before this work
# merges.
VI.scalartype(::Type{<:AbstractNamedDimsArray{T}}) where {T} = T
function VI.zerovector(x::AbstractNamedDimsArray, ::Type{S}) where {S <: Number}
    return fill!(similar(x, S), zero(S))
end
VI.scale(x::AbstractNamedDimsArray, α::Number) = x * α
VI.scale!!(x::AbstractNamedDimsArray, α::Number) = x * α
VI.scale!!(::AbstractNamedDimsArray, x::AbstractNamedDimsArray, α::Number) = x * α
function VI.add!!(
        y::AbstractNamedDimsArray, x::AbstractNamedDimsArray, α::Number, β::Number
    )
    return x * α + y * β
end
VI.inner(x::AbstractNamedDimsArray, y::AbstractNamedDimsArray) = (conj(x) * y)[]
# --------------------------------------------------------------------------------------

# Lowest eigenpair of the effective Hamiltonian operator `H_eff` acting on the named tensor
# `T` (via `NamedDimsArrays.apply`), using `KrylovKit.eigsolve`.
function eigsolve_named(H_eff, T, which)
    vals, vecs = eigsolve(x -> NDA.apply(H_eff, x), T, 1, which; ishermitian = true)
    return real(vals[1]), vecs[1]
end

function region_update_nsite!(::Val{2}, alg::TwoSiteEigsolve, qf, env, region)
    v1, v2 = region
    ln = only(linknames(qf.ket, v1 => v2))
    rows = collect(setdiff(dimnames(qf.ket[v1]), [ln]))

    T = qf.ket[v1] * qf.ket[v2]
    H_eff = effective_hamiltonian(qf, env, region)
    energy, T_opt = eigsolve_named(H_eff, T, alg.which)

    U, S, Vt = svd_split(T_opt, rows; alg.trunc)
    bond = only(setdiff(dimnames(U), rows))
    # Center on `v2` (the direction the sweep moves): `v1` is left isometric.
    setindex_preserve_graph!(qf.ket, replacedimnames(U, bond => ln), v1)
    setindex_preserve_graph!(qf.ket, replacedimnames(S * Vt, bond => ln), v2)
    return qf, energy
end

# ============================ Layered DMRG algorithm ====================================
#
# Three layers mirroring `beliefpropagation` / `apply_operators`: the outer sweep loop
# (`DMRGProblem` / `DMRGAlgorithm`), one sweep over regions (`DMRGSweepProblem` /
# `DMRGSweepAlgorithm`), and the per-region update strategy (`DMRGUpdateAlgorithm`, default
# `TwoSiteEigsolve`). The iterate is the ket tensor network; the operator and ket→bra name
# maps are fixed problem data.
#
# Environments are maintained incrementally across a sweep. Each sweep starts by gauging the
# ket onto the first region and building the exact tree environment once (O(N)); each region
# step then moves the orthogonality center along the tree path to the region (refreshing the
# forward message on each walked edge) and refreshes the message leaving the region after the
# local solve. So a sweep costs O(N), not O(N^2).

# Default sweep plan: every tree edge as a 2-site region, both directions (a back-and-forth
# sweep), via `forest_cover_edge_sequence`.
function default_dmrg_regions(ket; kwargs...)
    return [[src(e), dst(e)] for e in forest_cover_edge_sequence(ket; kwargs...)]
end

# Recompute the environment message on `v → w` from the (current) subtree on `v`'s side:
# `qf[v]` contracted with the incoming messages from `v`'s other neighbors. Used to refresh
# a message after the tensor at `v` has been re-gauged or updated.
function set_environment_message!(env, qf, v, w)
    incoming = incoming_subtree_messages(env, qf, v, w)
    message = contract_network([[qf[v]]; incoming])
    env[edgetype(qf)(v, w)] = environment_operator(message, qf.link_index_map)
    return env
end

# Walk the orthogonality center from its current vertex to `target` along the tree path,
# QR-gauging each edge and refreshing the forward environment message so the env stays
# exact. No-op when the center is already at `target`.
function move_center!(state, qf, target)
    for e in edge_path(state.iterate, state.center, target)
        gauge_move!(state.iterate, src(e), dst(e))
        set_environment_message!(state.env, qf, src(e), dst(e))
    end
    state.center = target
    return state
end

# === Layer 2: one sweep over regions ===

struct DMRGSweepProblem{Operator, Regions, LinkMap} <: AI.Problem
    operator::Operator
    regions::Regions
    link_index_map::LinkMap
end

@kwdef struct DMRGSweepAlgorithm{
        UpdateAlgorithm, StoppingCriterion <: AI.StoppingCriterion,
    } <: AI.Algorithm
    update_algorithm::UpdateAlgorithm = TwoSiteEigsolve()
    stopping_criterion::StoppingCriterion
end

@kwdef mutable struct DMRGSweepState{
        Iterate, Env, V, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: AI.State
    iterate::Iterate
    env::Env
    center::V
    energy::Float64 = 0.0
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function quadraticformnetwork(problem::DMRGSweepProblem, ket)
    return QuadraticFormNetwork(ket, problem.operator, problem.link_index_map)
end

# Gauge the ket onto the first region's center and build the exact tree environment from
# scratch. Run once at the start of each sweep; per-step updates keep it current.
function prepare_sweep(problem::DMRGSweepProblem, iterate)
    center = first(first(problem.regions))
    ket = orthogonalize(iterate, center)
    qf = quadraticformnetwork(problem, ket)
    env = quadratic_form_environments(qf; root_vertex = _ -> center)
    return ket, env, center
end

function AI.initialize_state(
        problem::DMRGSweepProblem, algorithm::DMRGSweepAlgorithm;
        iterate, iteration::Int = 0
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    ket, env, center = prepare_sweep(problem, iterate)
    return DMRGSweepState(; iterate = ket, env, center, iteration, stopping_criterion_state)
end

# The iterate, environment, and center are carried from the previous sweep (built once in
# `initialize_state`). Reset only the per-sweep counters and move the center back to the
# first region's start — a no-op for the default round-trip plan, an incremental walk for a
# custom plan that ends elsewhere. The environment is never rebuilt from scratch.
function AI.initialize_state!(
        problem::DMRGSweepProblem, algorithm::DMRGSweepAlgorithm,
        state::DMRGSweepState; iteration::Int = 0
    )
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    target = first(first(problem.regions))
    if state.center != target
        move_center!(state, quadraticformnetwork(problem, state.iterate), target)
    end
    return state
end

function AI.step!(
        problem::DMRGSweepProblem, algorithm::DMRGSweepAlgorithm, state::DMRGSweepState
    )
    region = problem.regions[state.iteration]
    v1, v2 = region[1], region[2]
    qf = quadraticformnetwork(problem, state.iterate)
    move_center!(state, qf, v1)
    qf, energy = region_update!(algorithm.update_algorithm, qf, state.env, region)
    # The split left the center on `v2` and finalized `v1`; refresh the message it emits.
    set_environment_message!(state.env, qf, v1, v2)
    state.center = v2
    state.energy = energy
    return state
end

# === Layer 1: outer sweep loop ===

struct DMRGProblem{Operator, LinkMap} <: AI.Problem
    operator::Operator
    link_index_map::LinkMap
end

# `sweep_update_algorithm(n)` returns the per-region update algorithm for sweep `n`, so the
# truncation (or any update-algorithm parameter) can vary by sweep.
@kwdef struct DMRGAlgorithm{
        Regions, SweepUpdateAlgorithm, StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.NestedAlgorithm
    regions::Regions
    sweep_update_algorithm::SweepUpdateAlgorithm
    stopping_criterion::StoppingCriterion
end

function sweep_algorithm(algorithm::DMRGAlgorithm, n::Int)
    return DMRGSweepAlgorithm(;
        update_algorithm = algorithm.sweep_update_algorithm(n),
        stopping_criterion = AI.StopAfterIteration(length(algorithm.regions))
    )
end

@kwdef mutable struct DMRGState{
        Substate <: AI.State, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: AIE.NestedState
    substate::Substate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

# The current energy lives on the substate; expose it on the outer state too.
energy(state::DMRGState) = state.substate.energy

function AI.initialize_state(
        problem::DMRGProblem, algorithm::DMRGAlgorithm; iterate, iteration::Int = 0
    )
    subproblem = DMRGSweepProblem(
        problem.operator, algorithm.regions, problem.link_index_map
    )
    substate = AI.initialize_state(subproblem, sweep_algorithm(algorithm, 1); iterate)
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return DMRGState(; iteration, stopping_criterion_state, substate)
end

function AI.initialize_state!(
        problem::DMRGProblem, algorithm::DMRGAlgorithm, state::DMRGState;
        iteration::Int = 0
    )
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    return state
end

function AIE.initialize_subsolve(
        problem::DMRGProblem, algorithm::DMRGAlgorithm, state::DMRGState
    )
    subproblem = DMRGSweepProblem(
        problem.operator, algorithm.regions, problem.link_index_map
    )
    return subproblem, sweep_algorithm(algorithm, state.iteration), state.substate
end

function AI.finalize_state!(::DMRGProblem, ::DMRGAlgorithm, state::DMRGState)
    return state.iterate, energy(state)
end

# === Energy-based convergence ===

@kwdef struct StopWhenEnergyConverged <: AI.StoppingCriterion
    tol::Float64
end

@kwdef mutable struct StopWhenEnergyConvergedState <: AI.StoppingCriterionState
    delta::Float64 = Inf
    at_iteration::Int = -1
    previous_energy::Float64 = NaN
end

function AI.initialize_state(
        ::AI.Problem, ::AI.Algorithm, ::StopWhenEnergyConverged; iterate
    )
    return StopWhenEnergyConvergedState()
end

function AI.initialize_state!(
        ::AI.Problem, ::AI.Algorithm, ::StopWhenEnergyConverged,
        st::StopWhenEnergyConvergedState
    )
    st.delta = Inf
    st.previous_energy = NaN
    return st
end

function AI.is_finished!(
        problem::AI.Problem, algorithm::AI.Algorithm, state::AI.State,
        c::StopWhenEnergyConverged, st::StopWhenEnergyConvergedState
    )
    current_energy = energy(state)
    previous_energy = st.previous_energy
    st.previous_energy = current_energy
    # No previous-sweep energy to compare against before the first sweep completes.
    state.iteration == 0 && return false
    st.delta = abs(current_energy - previous_energy)
    if AI.is_finished(problem, algorithm, state, c, st)
        st.at_iteration = state.iteration
        return true
    end
    return false
end

function AI.is_finished(
        ::AI.Problem, ::AI.Algorithm, ::AI.State,
        c::StopWhenEnergyConverged, st::StopWhenEnergyConvergedState
    )
    return st.delta < c.tol
end

# === Top-level entry point ===

select_dmrg_stopping_criterion(c::AI.StoppingCriterion) = c
function select_dmrg_stopping_criterion(::Nothing)
    return throw(
        ArgumentError(
            "`stopping_criterion` must be specified, e.g.\n" *
                "  `stopping_criterion = (; maxsweeps = 10)`,\n" *
                "  `stopping_criterion = (; maxsweeps = 10, tol = 1.0e-10)`, or\n" *
                "  `stopping_criterion = AI.StopAfterIteration(10) | StopWhenEnergyConverged(1.0e-10)`."
        )
    )
end
function select_dmrg_stopping_criterion(kwargs::NamedTuple)
    return select_dmrg_stopping_criterion(; kwargs...)
end
function select_dmrg_stopping_criterion(; maxsweeps = nothing, tol = nothing, kwargs...)
    if !isempty(kwargs)
        throw(
            ArgumentError(
                "Unrecognized `stopping_criterion` kwargs: $(keys(kwargs)). " *
                    "Supported: `maxsweeps`, `tol`."
            )
        )
    end
    if isnothing(maxsweeps) && isnothing(tol)
        throw(ArgumentError("At least one of `maxsweeps` or `tol` must be specified."))
    end
    criterion = nothing
    if !isnothing(maxsweeps)
        criterion = AI.StopAfterIteration(maxsweeps)
    end
    if !isnothing(tol)
        converged = StopWhenEnergyConverged(; tol)
        criterion = isnothing(criterion) ? converged : criterion | converged
    end
    return criterion
end

"""
    dmrg(operator, ket; stopping_criterion, kwargs...) -> (ket, energy)

Two-site DMRG for the ground state of `operator` (a [`TensorNetworkOperator`](@ref) on the
same tree as `ket`) starting from the ket tensor network `ket`. Returns the optimized ket
and its energy.

Keyword arguments:

  - `stopping_criterion` (required): an `AI.StoppingCriterion`, or a `NamedTuple` accepting
    `maxsweeps` (cap on back-and-forth passes) and/or `tol` (stop once the energy change
    between consecutive sweeps drops below `tol`), e.g. `(; maxsweeps = 10, tol = 1.0e-10)`.
  - `trunc`: a function `n -> trunc` giving the SVD-split truncation for sweep `n`. Cannot
    be combined with an explicit `alg`.
  - `regions`: sweep plan, a vector of vertex-vector regions (default: every tree edge in
    both directions).
  - `link_index_map`: ket→bra link name map (default: a fresh name per ket link).
  - `alg`: per-region update algorithm (fixed across sweeps). Other keyword arguments
    (e.g. `which`) are forwarded to the default `TwoSiteEigsolve`.
"""
function dmrg(
        operator::TensorNetworkOperator, ket;
        stopping_criterion = nothing,
        trunc = nothing,
        regions = default_dmrg_regions(ket),
        link_index_map = Dict(
            ln => randname(ln) for e in edges(ket) for ln in linknames(ket, e)
        ),
        alg = nothing,
        kwargs...
    )
    sweep_update_algorithm = if isnothing(alg)
        sweep_trunc = isnothing(trunc) ? Returns(nothing) : trunc
        n -> select_algorithm(
            region_update!,
            nothing,
            (ket,);
            trunc = sweep_trunc(n),
            kwargs...
        )
    else
        isnothing(trunc) || throw(
            ArgumentError(
                "`trunc` cannot be combined with an explicit `alg`; set it on `alg`."
            )
        )
        Returns(select_algorithm(region_update!, alg, (ket,)))
    end
    algorithm = DMRGAlgorithm(;
        regions, sweep_update_algorithm,
        stopping_criterion = select_dmrg_stopping_criterion(stopping_criterion)
    )
    problem = DMRGProblem(operator, link_index_map)
    return AI.solve(problem, algorithm; iterate = copy(ket))
end
