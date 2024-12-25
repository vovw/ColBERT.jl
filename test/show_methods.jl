# Define mock structs at top level
struct MockIndexer
    config::ColBERTConfig
    collection::Vector{String}
end

function Base.show(io::IO, ::MIME"text/plain", indexer::MockIndexer)
    print(io, "ColBERT Indexer:\n")
    print(io, "  collection size: $(length(indexer.collection)) documents\n")
    print(io, "  checkpoint: $(indexer.config.checkpoint)\n")
    print(io, "  index path: $(indexer.config.index_path)\n")
end

struct MockSearcher
    config::ColBERTConfig
    doclens::Vector{Int}
    centroids::Matrix{Float32}
end

function Base.show(io::IO, ::MIME"text/plain", searcher::MockSearcher)
    print(io, "ColBERT Searcher:\n")
    print(io, "  checkpoint: $(searcher.config.checkpoint)\n")
    print(io, "  index path: $(searcher.config.index_path)\n")
    print(io, "  embeddings:\n")
    print(io, "    total: $(sum(searcher.doclens))\n")
    print(io, "    centroids: $(size(searcher.centroids,2))\n")
end

# Now the actual tests
@testset "show methods" begin
    mktempdir() do dir
        config = ColBERTConfig(
            checkpoint = "dummy-checkpoint",
            index_path = dir,
            collection = ["doc1", "doc2"]
        )

        str = sprint(show, MIME("text/plain"), config)
        @test occursin("    checkpoint: dummy-checkpoint", str)
        @test occursin("    path: $dir", str)
        @test occursin("    collection: 2 documents", str)

        mock_indexer = MockIndexer(config, ["doc1", "doc2"])
        str = sprint(show, MIME("text/plain"), mock_indexer)
        @test occursin("  checkpoint: dummy-checkpoint", str)
        @test occursin("  index path: $dir", str)
        @test occursin("  collection size: 2", str)

        mock_searcher = MockSearcher(
            config,
            [10, 20],
            rand(Float32, 128, 5)
        )

        str = sprint(show, MIME("text/plain"), mock_searcher)
        @test occursin("  checkpoint: dummy-checkpoint", str)
        @test occursin("  index path: $dir", str)
        @test occursin("    total: 30", str)
        @test occursin("    centroids: 5", str)
    end
end
