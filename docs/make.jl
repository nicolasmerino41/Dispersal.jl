using Documenter, Dispersal
using Dispersal: rule, neighbors, build_dispersal_kernel, pressure

makedocs(
    modules = [Dispersal],
    doctest = false,
    clean = false,
    sitename = "Dispersal.jl",
    format = :html,
    pages = Any[
        "Introduction" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/rafaqz/Dispersal.jl.git",
    osname = "linux",
    julia = "0.6",
    target = "build",
    deps = nothing,
    make = nothing
)
