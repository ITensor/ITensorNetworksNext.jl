using NamedDimsArrays: AbstractNamedDimsArrayStyle

# Lazy broadcasting.
struct LazyNamedDimsArrayStyle <: AbstractNamedDimsArrayStyle{Any} end
function Broadcast.broadcasted(::LazyNamedDimsArrayStyle, f, as...)
    return error("Arbitrary broadcasting not supported for LazyNamedDimsArray.")
end
# Linear operations.
Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(+), a1, a2) = a1 + a2
Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(-), a1, a2) = a1 - a2
Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(*), c::Number, a) = c * a
Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(*), a, c::Number) = a * c
Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(*), a::Number, b::Number) = a * b
Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(/), a, c::Number) = a / c
Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(-), a) = -a
