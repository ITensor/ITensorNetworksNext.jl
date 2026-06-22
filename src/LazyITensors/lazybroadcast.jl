# Lazy broadcasting.
struct LazyITensorStyle <: Base.Broadcast.AbstractArrayStyle{Any} end
function Broadcast.broadcasted(::LazyITensorStyle, f, as...)
    return error("Arbitrary broadcasting not supported for LazyITensor.")
end
# Linear operations.
Broadcast.broadcasted(::LazyITensorStyle, ::typeof(+), a1, a2) = a1 + a2
Broadcast.broadcasted(::LazyITensorStyle, ::typeof(-), a1, a2) = a1 - a2
Broadcast.broadcasted(::LazyITensorStyle, ::typeof(*), c::Number, a) = c * a
Broadcast.broadcasted(::LazyITensorStyle, ::typeof(*), a, c::Number) = a * c
Broadcast.broadcasted(::LazyITensorStyle, ::typeof(*), a::Number, b::Number) = a * b
Broadcast.broadcasted(::LazyITensorStyle, ::typeof(/), a, c::Number) = a / c
Broadcast.broadcasted(::LazyITensorStyle, ::typeof(-), a) = -a
