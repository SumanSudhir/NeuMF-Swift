import TensorFlow
import Foundation

public struct NeuMF: Module {

    public typealias Scalar = Float

    @noDerivative public let num_users: Int
    @noDerivative public let num_items: Int
    @noDerivative public let mf_dim: Int
    @noDerivative public let mf_reg: Scalar
    @noDerivative public var mlp_layer_sizes : [Int] = [64,32,16,8]
    @noDerivative public var mlp_layer_regs: [Scalar] = [0,0,0,0]

    public var mf_user_embed: Embedding<Scalar>
    public var mf_item_embed: Embedding<Scalar>
    public var mlp_user_embed: Embedding<Scalar>
    public var mlp_item_embed: Embedding<Scalar>
    public var dense1: Dense<Scalar>
    public var dense2: Dense<Scalar>
    public var dense3: Dense<Scalar>
    public var final_dense: Dense<Scalar>


    public init(
        num_users: Int,
        num_items: Int,
        mf_dim: Int,
        mf_reg: Float,
        mlp_layer_sizes: [Int],
        mlp_layer_regs: [Float]
    ) {
        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.mf_reg = mf_reg
        self.mlp_layer_sizes = mlp_layer_sizes
        self.mlp_layer_regs = mlp_layer_regs

        // precondition(self.mlp_layer_sizes[0]%2 == 0, "u dummy, mlp_layer_sizes[0] % 2 != 0")
        // precondition(self.mlp_layer_sizes.count == self.mlp_layer_regs.count, "u dummy, layer_sizes != layer_regs!")

        //TODO: regularization
        self.mf_user_embed = Embedding<Scalar>(vocabularySize: self.num_users, embeddingSize: self.mf_dim)
        self.mf_item_embed = Embedding<Scalar>(vocabularySize: self.num_items, embeddingSize: self.mf_dim)
        self.mlp_user_embed = Embedding<Scalar>(vocabularySize: self.num_users, embeddingSize: self.mlp_layer_sizes[0]/2)
        self.mlp_item_embed = Embedding<Scalar>(vocabularySize: self.num_items, embeddingSize: self.mlp_layer_sizes[0]/2)

        //TODO: Extend it for n layers by using for loop
        //Currently only for 3 layers
        dense1 = Dense(inputSize: self.mlp_layer_sizes[0], outputSize: self.mlp_layer_sizes[1], activation: relu)
        dense2 = Dense(inputSize: self.mlp_layer_sizes[1], outputSize: self.mlp_layer_sizes[2], activation: relu)
        dense3 = Dense(inputSize: self.mlp_layer_sizes[2], outputSize: self.mlp_layer_sizes[3], activation: relu)
        final_dense = Dense(inputSize: (self.mlp_layer_sizes[3] + self.mf_dim), outputSize: 1)
    }
        @differentiable
        public func callAsFunction(_ input: Tensor<Int32>) -> Tensor<Scalar>{
            let user_indices = input.unstacked(alongAxis:1)[0]
            let item_indices = input.unstacked(alongAxis:1)[1]

            let user_embed_mlp = self.mlp_user_embed(user_indices)
            let item_embed_mlp = self.mlp_item_embed(item_indices)
            let user_embed_mf = self.mf_user_embed(user_indices)
            let item_embed_mf = self.mf_item_embed(item_indices)

            // let mf_vector = matmul(user_embed_mf,item_embed_mf)
            let mf_vector = user_embed_mf*item_embed_mf
            var mlp_vector = user_embed_mlp.concatenated(with:item_embed_mlp,alongAxis:-1)
            //
            // print(mlp_vector.shape)
            mlp_vector = mlp_vector.sequenced(through: dense1, dense2, dense3)
            let vector = mlp_vector.concatenated(with:mf_vector,alongAxis:-1)

            return final_dense(vector)
            // return mf_vector
        }
    // }
}
