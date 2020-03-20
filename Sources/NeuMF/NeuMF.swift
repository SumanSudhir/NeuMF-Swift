import TensorFlow
import Foundation

public struct NeuMF: Module {

    public typealias Scalar = Float
    // Total number of users
    @noDerivative public let numUsers: Int
    // Total number of items
    @noDerivative public let numItems: Int
    //Embedding size of MF model
    @noDerivative public let mfDim: Int
    //Regularization for MF embeddings
    @noDerivative public let mfReg: Scalar
    // MLP layers: Note that the first layer is the concatenation of user and item embeddings
    // So mlpLayerSizes[0]/2 is the embedding size."
    @noDerivative public var mlpLayerSizes : [Int] = [64,32,16,8]
    //Regularization for MLP layer
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

        precondition(mlpLayerSizes[0]%2 == 0, "Input of first MLP layers must be multiple of 2")
        precondition(mlpLayerSizes.count == mlpLayerRegs.count, "Size of MLP layers and MLP reqularization must be equal")

        //TODO: regularization
        //Embedding Layer
        // User MF embedding
        self.mfUserEmbed = Embedding<Scalar>(vocabularySize: self.numUsers, embeddingSize: self.mfDim)
        // Item MF embedding
        self.mfItemEmbed = Embedding<Scalar>(vocabularySize: self.numItems, embeddingSize: self.mfDim)
        // User MLP embedding
        self.mlpUserEmbed = Embedding<Scalar>(vocabularySize: self.numUsers, embeddingSize: self.mlpLayerSizes[0]/2)
        // Item MLP embedding
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
            // Extracting user and item from dataset
            let userIndices = input.unstacked(alongAxis:1)[0]
            let itemIndices = input.unstacked(alongAxis:1)[1]

            // MLP part
            let userEmbedMlp = self.mlpUserEmbed(userIndices)
            let itemEmbedMlp = self.mlpItemEmbed(itemIndices)

            // MF part
            let userEmbedMf = self.mfUserEmbed(userIndices)
            let itemEmbedMf = self.mfItemEmbed(itemIndices)

            //Concatenate MF and MLP parts
            let mfVector = userEmbedMf*itemEmbedMf
            var mlpVector = userEmbedMlp.concatenated(with:itemEmbedMlp,alongAxis:-1)

            // Final prediction layer
            mlpVector = mlpVector.sequenced(through: dense1, dense2, dense3)
            let vector = mlpVector.concatenated(with:mfVector,alongAxis:-1)

            return finalDense(vector)
        }
}
