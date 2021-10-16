using MSM
using Documenter

DocMeta.setdocmeta!(MSM, :DocTestSetup, :(using MSM); recursive=true)

makedocs(;
    modules=[MSM],
    authors="banachtech <balaji@banach.tech> and contributors",
    repo="https://github.com/banachtech/MSM.jl/blob/{commit}{path}#{line}",
    sitename="MSM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://banachtech.github.io/MSM.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/banachtech/MSM.jl",
)
