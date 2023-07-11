module Cyclicity

using DataFrames, LinearAlgebra, StatsBase, CSV, Ellipses

# Remove linear trend or DC bias
# ------------------------------------------------
"""Subtract mean value of a vector from itself"""
mean_center(vec::Vector{<:Number}) :: Vector = vec .- mean(vec)

"""Mean center columns of a data frame"""
mean_center(df::DataFrame) :: DataFrame = mapcols(mean_center ∘ Array, df)

"""Linearly match ends of vector"""
function match_ends(vec::Vector{<:Number})
    t = range(first(vec), stop=last(vec), length=length(vec))
    return vec .- t 
end

"""Adjust dataframe columns to have same start and end values"""
match_ends(df::DataFrame) :: DataFrame = mapcols(match_ends ∘ Array, df)

export mean_center, match_ends

# Normalization options
# ------------------------------------------------
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
# ------------------------------------------------
"""Cyclically differentiate a vector"""
cycdiff(v::Vector{<:Number})::Vector = diff(vcat(v[end], v))

"""Compute areavalue between two vectors (makes assumptions)"""
areaval(x::Vector{<:Number}, y::Vector{<:Number}) :: Number = (x⋅cycdiff(y) - y⋅cycdiff(x))/2

export cycdiff, areaval

# Loop to create lead matrix
# ------------------------------------------------

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

# Structures to facilitate some typical data analysis routines
# ------------------------------------------------
struct Scan
	name :: String
	task :: String
	run :: String
	data :: DataFrame
end

struct CyData
	name :: String
	task :: String
	run :: String
	norm :: Symbol
	lm :: Matrix{<:Number}
	lambdas :: Vector{<:Number}
	phases :: Matrix{<:Number}
end

struct ElData
	name :: String
	task :: String
	run :: String
	norm :: Symbol
	phase :: Vector{<:Number}
	pnum :: Integer
	ell :: EllipseQform
end


function parsename(name::AbstractString)
	"""Expects name to be of the form `rfMRI_REST1_LR.csv`"""
	modality, task, encoding = split(name, "_")
	run, _ = split(encoding, ".")
	return task, run 
end

readcsv(csvfile::String)= CSV.read(csvfile, DataFrame, ntasks=6)

function readfolder(folder::String) :: Array{Scan}
	"""Expects in the form basefolder: `CSV_1200` and folder:`100206` """
	csvfiles= csvfilesin(folder)
	params = map(parsename, csvfiles)
	filepaths = map(x->joinpath(basefolder, folder, x), csvfiles)
	[Scan(folder, task, run, readcsv(fp)) for ((task, run), fp) in zip(params, filepaths)]
end

function Scan(filename::String) :: Scan
	_, name, fname = split(filename, "/")
	task, run = parsename(fname)
	Scan(name, task, run, readcsv(filename))
end
export Scan

function CyData(s::Scan, norm::Function) :: CyData
    n, t, r = s.name, s.task, s.run
    lm = s.data |> (make_lead_matrix ∘ Matrix ∘ norm ∘ mean_center ∘ match_ends)
    evals, evecs = eigen(lm, sortby= λ -> -abs(λ))
    CyData(n, t, r, Symbol(norm), lm, evals, evecs)
end
export CyData


function ElData(s::CyData, pnum::Integer) :: ElData
	phase = s.phases[:, pnum]
	qell = fit_ellipse(phase)
	n, t, r, norm = s.name, s.task, s.run, s.norm
	ElData(n, t, r, norm, phase, pnum, qell)
end

export ElData
end
