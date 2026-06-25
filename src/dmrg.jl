import .AlgorithmsInterfaceExtensions as AIE
import AlgorithmsInterface as AI
using Base: @kwdef
using Graphs: dst, edges, edgetype, src
using ITensorBase:
    ITensorBase as ITB, dimnames, operator, replacedimnames, state, uniquename
using KrylovKit: eigsolve
using MatrixAlgebraKit: svd_trunc
using NamedGraphs.GraphsExtensions: edge_path

# ============================ Top-level entry point =====================================

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
            ln => uniquename(ln) for e in edges(ket) for ln in linknames(ket, e)
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

# Default sweep plan: every tree edge as a 2-site region, both directions (a back-and-forth
# sweep), via `forest_cover_edge_sequence`.
function default_dmrg_regions(ket; kwargs...)
    return [[src(e), dst(e)] for e in forest_cover_edge_sequence(ket; kwargs...)]
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

# Recompute the environment message on `v → w` from the (current) subtree on `v`'s side:
# `qf[v]` contracted with the incoming messages from `v`'s other neighbors. Used to refresh
# a message after the tensor at `v` has been re-gauged or updated.
function set_environment_message!(env, qf, v, w)
    incoming = incoming_subtree_messages(env, qf, v, w)
    message = contract_network([[qf[v]]; incoming])
    env[edgetype(qf)(v, w)] = environment_operator(message, qf.link_index_map)
    return env
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

svd_split(T, rows; trunc) = svd_trunc(T, rows; trunc)

# Lowest eigenpair of the effective Hamiltonian operator `H_eff` acting on the named tensor
# `T` (via `ITensorBase.apply`), using `KrylovKit.eigsolve`.
function eigsolve_named(H_eff, T, which)
    vals, vecs = eigsolve(x -> ITB.apply(H_eff, x), T, 1, which; ishermitian = true)
    return real(vals[1]), vecs[1]
end

# The effective Hamiltonian for a `region` (a vector of vertices) is the projected operator
# obtained by contracting the operator tensors on the region with the incoming environment
# messages on the region's boundary. Contracting those gives a single tensor whose codomain
# (output) names are the region's bra names and whose domain (input) names are its ket names
# — i.e. exactly an ITensor operator. So `effective_hamiltonian` returns that operator, and
# its action on a region ket tensor `T` is `ITensorBase.apply(H, T)` (which contracts the ket
# names and renames the resulting bra names back to ket names).

"""
    effective_hamiltonian(qf::QuadraticFormNetwork, env, region) -> ITensorOperator

Effective (projected) Hamiltonian for `region` (a vector of vertices) as an ITensor
operator: its domain (input) names are the region's ket names and its codomain (output)
names are the matching bra names. Apply it to a region ket tensor `T` with
`ITensorBase.apply`. The environment `env` is a `MessageCache` of
[`quadratic_form_environments`](@ref).
"""
function effective_hamiltonian(qf::QuadraticFormNetwork, env, region)
    operators = [operator_tensor(qf, v) for v in region]
    boundary = [state(m) for m in incoming_edge_data(env, region)]
    h = contract_network([operators; boundary])
    sitemap = site_index_map(qf.operator)
    ketnames = [
        n for n in dimnames(h) if haskey(sitemap, n) || haskey(qf.link_index_map, n)
    ]
    branames = [bra_name_map(qf)[n] for n in ketnames]
    return operator(h, branames, ketnames)
end

# ============================ Energy-based convergence ==================================

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
