module Cyclicity

using DataFrames, LinearAlgebra, StatsBase, DataStructures
# Remove linear trend or DC bias

mean_center(vec::Vector{<:Number}) :: Vector = vec .- mean(vec)
"""Mean center columns of a data frame"""
mean_center(df::DataFrame) :: DataFrame = mapcols(mean_center ∘ Array, df)

"""Adjust dataframe columns to have same start and end values"""
match_ends(df::DataFrame) :: DataFrame = mapcols(match_ends ∘ Array, df)
"""Linearly match ends of vector"""
function match_ends(vec::Vector{<:Number})
    t = range(first(vec), stop=last(vec), length=length(vec))
    return vec .- t 
end

export mean_center, match_ends

# Normalization options
"""Normalize a vector by its standard deviation"""
std_norm(vec::Vector{<:Number}) :: Vector = vec ./ std(vec)
"""Normalize colums of a dataframe by their standard deviation"""
std_norm(df::DataFrame) :: DataFrame = mapcols(std_norm ∘ Array, df)

"""Normalize a vector by its Euclidean norm"""
quad_norm(vec::Vector{<:Number}) :: Vector = vec ./ norm(vec)
"""Normalize columns of dataframe by Euclidean norms"""
quad_norm(df::DataFrame) :: DataFrame = mapcols(quad_norm ∘ Array, df)

"""Normalize a vector by its total variation norm"""
totvar_norm(vec::Vector{<:Number}) :: Vector = vec ./ norm(cycdiff(vec))
"""Normalize columns of dataframe by total variation"""
totvar_norm(df::DataFrame) :: DataFrame = mapcols(totvar_norm ∘ Array, df)

export std_norm, quad_norm, totvar_norm

# Differentiation & Green's integral 
"""Cyclically differentiate a vector"""
cycdiff(v::Vector{<:Number})::Vector = diff(vcat(v[end], v))

"""Compute areavalue between two vectors (makes assumptions)"""
areaval(x::Vector{<:Number}, y::Vector{<:Number}) :: Number = (x⋅cycdiff(y) - y⋅cycdiff(x))/2

"""Cumulatively compute the parametric area between two time series"""
function cumul_area(pair::Tuple{Vector{T}, Vector{T}}) :: Vector{T} where T
	cumul_area(pair[1], pair[2])
end

"""Cumulatively compute the parametric area between two time series"""
function cumul_area(x::Vector{T}, y::Vector{T}) :: Vector{T} where T
	@assert length(x) == length(y)
	z = similar(x)
	z[1] = 0
	for n=2:length(z)
		z[n] = z[n-1] + (x[1]-x[n])*(y[n-1]-y[n]) - (y[1]-y[n])*(x[n-1]-x[n])
	end
	return z/2
end
export cycdiff, areaval, cumul_area

# Loop to create lead matrix
"""Make a lead matrix from the columns of a dataframe"""
make_lead_matrix(df::DataFrame) = make_lead_matrix(Matrix(df))

"""Function to make lead matrix from columns of a matrix"""
function make_lead_matrix(arr::Matrix{T}) where T
	_, N = size(arr)
	lmat = zeros(T, N, N)
	@inbounds for (i, icol) in enumerate(eachcol(arr))
		@inbounds @simd for j=i+1:N
				lmat[i, j] = areaval(icol[:], view(arr, :,j)[:])
				lmat[j, i] = -lmat[i, j]
		end
	end
	return lmat
end
export make_lead_matrix

function minmax_pairs(series::AbstractVector)
	mx = maximum(series)
	tser = [Inf, mx+10, series..., mx+10, Inf]
	pers = Deque{Tuple{Real, Integer, Real, Integer, Integer}}()
	stackx = Deque{Tuple{Real, Integer}}()
	stackn = Deque{Tuple{Real, Integer}}()
	up, opos = -1, 0
	
	for (pos, tp) in enumerate(tser[1:end-1])
		(up>0  && !isempty(stackx)) && 
			for i=1:sum(map(x -> first(x) <= tp, stackx))
				push!(pers, (pop!(stackn)..., pop!(stackx)..., pos))
			end
		(up<0 && !isempty(stackn)) &&
			for i=1:sum(map(x -> first(x) >= tp, stackn))
				push!(pers, (pop!(stackn)..., pop!(stackx)..., pos))
			end
		if (up*(tser[pos+1]-tp)<0)
			up > 0 ? push!(stackx, (tser[pos], pos-2)) :
				 push!(stackn, (tser[pos], pos-2))
			up = -up 
		end
		opos = pos
	end
	push!(pers, (pop!(stackn)..., NaN, 1, opos))
end
export minmax_pairs

end
