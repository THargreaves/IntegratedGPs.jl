using StaticArrays

export ShiftingTransition, ExpandingTransition
export quadratic_form, quadratic_form!

"""
An efficient sparse representation of the n x n transition matrix

    [ f₁   f₂    ⋯   f_{n-1}   f_n ]
    [  1    0    ⋯      0       0  ]
    [  0    1    ⋯      0       0  ]
    [  ⋮    ⋮    ⋱      ⋮       ⋮  ]
    [  0    0    ⋯      1       0  ]
"""
struct ShiftingTransition{T,V<:AbstractVector{T}} <: AbstractMatrix{T}
    f::V
end

# AbstractMatrix interface
Base.eltype(::ShiftingTransition{T}) where {T} = T
function Base.size(A::ShiftingTransition)
    n = length(A.f)
    return (n, n)
end
function Base.axes(A::ShiftingTransition)
    s = size(A)
    return (Base.OneTo(s[1]), Base.OneTo(s[2]))
end
Base.IndexStyle(::Type{<:ShiftingTransition}) = IndexCartesian()
function Base.getindex(A::ShiftingTransition, i::Integer, j::Integer)
    n = length(A.f)
    if j < 1 || j > n || i < 1 || i > n
        throw(BoundsError(A, (i, j)))
    end

    @inbounds begin
        if i == 1
            return A.f[j]
        elseif 2 <= i <= n
            return (j == (i - 1)) ? one(eltype(A.f)) : zero(eltype(A.f))
        end
    end
end

# Optimised linear algebra routines
function LinearAlgebra.mul!(
    y::AbstractVector{T}, A::ShiftingTransition{T}, x::AbstractVector{T}
) where {T}
    n = length(A.f)
    @assert length(x) == n
    @assert length(y) == n

    @inbounds begin
        # Store for later to allow case when x, y are the same array
        y1 = dot(A.f, x)
        # Recursive in reverse order to avoid overwriting x in the case x === y
        for i in n:-1:2
            y[i] = x[i - 1]
        end
        y[1] = y1
    end

    return y
end
function Base.:*(A::ShiftingTransition{T}, x::AbstractVector{T}) where {T}
    n = length(A.f)
    @assert length(x) == n
    y = similar(x)

    return mul!(y, A, x)
end
function Base.:*(A::ShiftingTransition{T,<:SVector{N,T}}, x::SVector{N,T}) where {N,T}
    y1 = dot(A.f, x)
    return SVector{N,T}((y1, x[SOneTo(N - 1)]...))
end

function quadratic_form!(
    C::Symmetric{T}, A::ShiftingTransition{T}, S::Symmetric{T}
) where {T}
    n = length(A.f)
    @assert size(S) == (n, n)
    @assert size(C) == (n, n)

    # Perform operations on parent arrays
    S = S.data
    C = C.data

    # Compute new components
    S11 = A.f' * S * A.f
    S1_ = @views S[1:(n - 1), 1:(n - 1)] * A.f[1:(n - 1)] + S[1:(n - 1), n] * A.f[n]

    # Fill C, which may be the same array as S
    @inbounds begin

        # Shift elements down and right
        for i in n:-1:2
            for j in n:-1:2
                C[i, j] = S[i - 1, j - 1]
            end
        end

        # Fill in first row and column
        C[1, 1] = S11
        C[2:n, 1] = S1_
        C[1, 2:n] = S1_'
    end

    return C
end
function quadratic_form(A::ShiftingTransition, S::Symmetric)
    n = length(A.f)
    @assert size(S) == (n, n)
    C = similar(S)

    return quadratic_form!(C, A, S)
end

"""
An efficient sparse representation of the (n + 1) x n transition matrix

    [ f₁   f₂    ⋯   f_{n-1}   f_n ]
    [  1    0    ⋯      0       0  ]
    [  0    1    ⋯      0       0  ]
    [  ⋮    ⋮    ⋱      ⋮       ⋮  ]
    [  0    0    ⋯      1       0  ]
    [  0    0    ⋯      0       1  ]
"""
struct ExpandingTransition{T,V<:AbstractVector{T}} <: AbstractMatrix{eltype(V)}
    f::V
end

# AbstractMatrix interface
Base.eltype(::ExpandingTransition{T}) where {T} = T
function Base.size(A::ExpandingTransition)
    n = length(A.f)
    return (n + 1, n)
end
function Base.axes(A::ExpandingTransition)
    s = size(A)
    return (Base.OneTo(s[1]), Base.OneTo(s[2]))
end
Base.IndexStyle(::Type{<:ExpandingTransition}) = IndexCartesian()
function Base.getindex(A::ExpandingTransition, i::Integer, j::Integer)
    n = length(A.f)
    if j < 1 || j > n || i < 1 || i > n + 1
        throw(BoundsError(A, (i, j)))
    end

    @inbounds begin
        if i == 1
            return A.f[j]
        elseif 2 <= i <= n + 1
            return (j == (i - 1)) ? one(eltype(A.f)) : zero(eltype(A.f))
        end
    end
end

# Optimised linear algebra routines
function LinearAlgebra.mul!(
    y::AbstractVector{T}, A::ExpandingTransition{T}, x::AbstractVector{T}
) where {T}
    n = length(A.f)
    @assert length(x) == n
    @assert length(y) == n + 1

    @inbounds begin
        # Store for later to allow case when x, y are the same array
        y1 = dot(A.f, x)
        # Recursive in reverse order to avoid overwriting x in the case x === y
        for i in (n + 1):-1:2
            y[i] = x[i - 1]
        end
        y[1] = y1
    end

    return y
end
function Base.:*(A::ExpandingTransition{T}, x::AbstractVector{T}) where {T}
    n = length(A.f)
    @assert length(x) == n
    y = similar(x, n + 1)

    return mul!(y, A, x)
end
function Base.:*(A::ExpandingTransition{T,<:SVector{N,T}}, x::SVector{N,T}) where {N,T}
    y1 = dot(A.f, x)
    return SVector{N,T}((y1, x...))
end

function quadratic_form!(
    C::Symmetric{T}, A::ExpandingTransition{T}, S::Symmetric{T}
) where {T}
    n = length(A.f)
    @assert size(S) == (n, n)
    @assert size(C) == (n + 1, n + 1)

    # Perform operations on parent arrays
    S = S.data
    C = C.data

    # Compute new components
    S11 = A.f' * S * A.f
    S1_ = S * A.f

    # Fill C, which may be the same array as S
    @inbounds begin

        # Shift elements down and right
        for i in (n + 1):-1:2
            for j in (n + 1):-1:2
                C[i, j] = S[i - 1, j - 1]
            end
        end

        # Fill in first row and column
        C[1, 1] = S11
        C[2:(n + 1), 1] = S1_
        C[1, 2:(n + 1)] = S1_'
    end

    return C
end
function quadratic_form(A::ExpandingTransition, S::Symmetric)
    n = length(A.f)
    @assert size(S) == (n, n)
    C = Symmetric(similar(S, n + 1, n + 1))

    return quadratic_form!(C, A, S)
end
