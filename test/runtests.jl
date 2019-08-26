using NamedPlus
using Test


@testset "unname with names" begin
    nda3 = NamedDimsArray{(:x, :y, :z)}(rand(10, 20, 30))
    nda2 = NamedDimsArray{(:x, :y)}(rand(10, 20))

    @test unname(nda3, (:x, :y, :z)) === parent(nda3)

    @test unname(nda3, (:z, :y, :x)) === PermutedDimsArray(parent(nda3), (3,2,1))

    @test unname(nda2, (:y, :x)) === transpose(parent(nda2))
end