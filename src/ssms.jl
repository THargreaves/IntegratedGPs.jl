export SSM

struct SSM{T<:AbstractFloat}
    A::Matrix{T}
    Q::Matrix{T}
    H::Matrix{T}

    function SSM(A::Matrix{T}, Q::Matrix{T}, H::Matrix{T}) where {T}
        return (
                   !(
                       length(size(A)) == 2 && length(size(Q)) == 2 && length(size(H)) == 2
                   ) && error(
                       "All provided matrices must be 2-dimensional; provided A: $(size(A)) and Q: $(size(Q)) and H: $(size(H))",
                   )
               ) ||
               (
                   !(size(Q)[1] == size(Q)[2]) &&
                   error("Q must be a square matrix; provided size is $(size(Q)).")
               ) ||
               (
                   !(size(A)[2] == size(Q)[1]) && error(
                       "A and Q must have compatible sizes; provided $(size(A)) and $(size(Q))",
                   )
               ) ||
               (
                   !(size(H)[2] == size(A)[1]) && error(
                       "H and A must have compatible sizes; provided $(size(H)) and $(size(A))",
                   )
               ) ||
               new{T}(A, Q, H)
    end
end