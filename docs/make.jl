push!(LOAD_PATH,"../src/")

using Documenter, FracFin

# makedocs()
makedocs(
    # options
    modules = [FracFin],

    format = :html,
    sitename = "FracFin.jl"
)