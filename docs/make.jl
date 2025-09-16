using ITensorNetworksNext: ITensorNetworksNext
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(
  ITensorNetworksNext, :DocTestSetup, :(using ITensorNetworksNext); recursive=true
)

include("make_index.jl")

makedocs(;
  modules=[ITensorNetworksNext],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="ITensorNetworksNext.jl",
  format=Documenter.HTML(;
    canonical="https://itensor.github.io/ITensorNetworksNext.jl",
    edit_link="main",
    assets=["assets/favicon.ico", "assets/extras.css"],
  ),
  pages=["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo="github.com/ITensor/ITensorNetworksNext.jl", devbranch="main", push_preview=true
)
