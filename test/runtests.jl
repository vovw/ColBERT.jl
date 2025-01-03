using ColBERT
using .Iterators
using JLD2
using JSON
using LinearAlgebra
using Logging
using Random
using Test

# turn off logging
logger = NullLogger()
global_logger(logger)

const INT_TYPES = [
    Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128]
const FLOAT_TYPES = [Float16, Float32, Float64]

# include("Aqua.jl")

# indexing operations
include("indexing/codecs/residual.jl")
include("indexing/collection_indexer.jl")

# modelling operations
include("modelling/tokenization/tokenizer_utils.jl")
include("modelling/embedding_utils.jl")

# search operations
include("searching.jl")
include("search/ranking.jl")

# show operations
include("show_methods.jl")

# utils
include("utils.jl")

# loaders and savers
include("loaders_and_savers.jl")
