module CyDataExt 

using Cyclicity, LinearAlgebra
using CSV, Ellipses, DataFrames 


struct Scan
	name :: String
	task :: String
	run :: String
	data :: DataFrame
end

struct CyData
	data :: Scan
	norm :: Symbol
	lm :: Matrix{<:Number}
	lambdas :: Vector{<:Number}
	phases :: Matrix{<:Number}
end

struct ElData
	data :: CyData
	pnum :: Integer
	phase :: Vector{<:Number}
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
	_..., name, fname = split(filename, "/")
	task, run = parsename(fname)
	Scan(name, task, run, readcsv(filename))
end


function CyData(s::Scan, norm::Function) :: CyData
    n, t, r = s.name, s.task, s.run
    lm = s.data |> (make_lead_matrix ∘ Matrix ∘ norm ∘ mean_center ∘ match_ends)
    evals, evecs = eigen(lm, sortby= λ -> -abs(λ))
    CyData(s, Symbol(norm), lm, evals, evecs)
end


function ElData(s::CyData, pnum::Integer) :: ElData
	phase = s.phases[:, pnum]
	qell = fit_ellipse(phase)
	ElData(s, pnum, phase, qell)
end

end
