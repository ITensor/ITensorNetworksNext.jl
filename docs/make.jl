using Documenter: Documenter, DocMeta, deploydocs, makedocs
using ITensorFormatter: ITensorFormatter
using ITensorNetworksNext: ITensorNetworksNext

DocMeta.setdocmeta!(
    ITensorNetworksNext, :DocTestSetup, :(using ITensorNetworksNext); recursive = true
)

ITensorFormatter.make_index!(pkgdir(ITensorNetworksNext))

makedocs(;
    modules = [ITensorNetworksNext],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "ITensorNetworksNext.jl",
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/ITensorNetworksNext.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"]
    ),
    pages = ["Home" => "index.md", "Reference" => "reference.md"]
)

deploydocs(;
    repo = "github.com/ITensor/ITensorNetworksNext.jl", devbranch = "main",
    push_preview = true
)
