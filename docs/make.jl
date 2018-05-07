# push!(LOAD_PATH,"../src/")

using Documenter, FracFin

# makedocs()

makedocs(
    # options
    modules = FracFin,
    format = :html,
    sitename = "FracFin.jl",
    authors = "Yann Calec",
    pages = Any[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Usage" => "usage.md"
    ]
)

# deploydocs(
#     deps = Deps.pip("mkdocs"),
#     repo = "github.com/yanncalec/FracFin.jl.git",
#     julia = "0.6"
# )