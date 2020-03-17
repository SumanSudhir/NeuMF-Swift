import TensorFlow
import Foundation

public struct NeuMF: Module {

    public typealias Scalar = Float

    @noDerivative public let numUsers: Int
    @noDerivative public let numItems: Int
    @noDerivative public let mfDim: Int
    @noDerivative public let mfReg: Scalar
    @noDerivative public var mlpLayerSizes : [Int] = [64,32,16,8]
    @noDerivative public var mlpLayerRegs: [Scalar] = [0,0,0,0]

    public var mfUserEmbed: Embedding<Scalar>
    public var mfItemEmbed: Embedding<Scalar>
    public var mlpUserEmbed: Embedding<Scalar>
    public var mlpItemEmbed: Embedding<Scalar>
    public var dense1: Dense<Scalar>
    public var dense2: Dense<Scalar>
    public var dense3: Dense<Scalar>
    public var finalDense: Dense<Scalar>


    public init(
        numUsers: Int,
        numItems: Int,
        mfDim: Int,
        mfReg: Float,
        mlpLayerSizes: [Int],
        mlpLayerRegs: [Float]
    ) {
        self.numUsers = numUsers
        self.numItems = numItems
        self.mfDim = mfDim
        self.mfReg = mfReg
        self.mlpLayerSizes = mlpLayerSizes
        self.mlpLayerRegs = mlpLayerRegs

        // precondition(self.mlpLayerSizes[0]%2 == 0, "u dummy, mlpLayerSizes[0] % 2 != 0")
        // precondition(self.mlpLayerSizes.count == self.mlpLayerRegs.count, "u dummy, layer_sizes != layer_regs!")

        //TODO: regularization
        self.mfUserEmbed = Embedding<Scalar>(vocabularySize: self.numUsers, embeddingSize: self.mfDim)
        self.mfItemEmbed = Embedding<Scalar>(vocabularySize: self.numItems, embeddingSize: self.mfDim)
        self.mlpUserEmbed = Embedding<Scalar>(vocabularySize: self.numUsers, embeddingSize: self.mlpLayerSizes[0]/2)
        self.mlpItemEmbed = Embedding<Scalar>(vocabularySize: self.numItems, embeddingSize: self.mlpLayerSizes[0]/2)

        //TODO: Extend it for n layers by using for loop
        //Currently only for 3 layers
        dense1 = Dense(inputSize: self.mlpLayerSizes[0], outputSize: self.mlpLayerSizes[1], activation: relu)
        dense2 = Dense(inputSize: self.mlpLayerSizes[1], outputSize: self.mlpLayerSizes[2], activation: relu)
        dense3 = Dense(inputSize: self.mlpLayerSizes[2], outputSize: self.mlpLayerSizes[3], activation: relu)
        finalDense = Dense(inputSize: (self.mlpLayerSizes[3] + self.mfDim), outputSize: 1)
    }
        @differentiable
        public func callAsFunction(_ input: Tensor<Int32>) -> Tensor<Scalar>{
            let userIndices = input.unstacked(alongAxis:1)[0]
            let itemIndices = input.unstacked(alongAxis:1)[1]

            let userEmbedMlp = self.mlpUserEmbed(userIndices)
            let itemEmbedMlp = self.mlpItemEmbed(itemIndices)
            let userEmbedMf = self.mfUserEmbed(userIndices)
            let itemEmbedMf = self.mfItemEmbed(itemIndices)

            let mfVector = userEmbedMf*itemEmbedMf
            var mlpVector = userEmbedMlp.concatenated(with:itemEmbedMlp,alongAxis:-1)

            mlpVector = mlpVector.sequenced(through: dense1, dense2, dense3)
            let vector = mlpVector.concatenated(with:mfVector,alongAxis:-1)

            return finalDense(vector)
        }
}
