import TensorFlow
import Foundation

public struct NeuMF: Layer {
    @noDerivative public let num_users: Int
    @noDerivative public let num_items: Int
    @noDerivative public let mf_dim: Int
    @noDerivative public let mf_reg: Float = 0.0
    @noDerivative public let mlp_layer_sizes : [Int] = [64,32,16,8]
    @noDerivative public let mlp_layer_regs: [Float] = [0,0,0,0]

    public init(
        num_users: Int,
        num_items: Int,
        mf_dim: Int,
        mf_reg: Float,
        mlp_layer_sizes: Int,
        mlp_layer_regs: Float
    ) {
        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.mf_reg = mf_reg
        self.mlp_layer_sizes = mlp_layer_sizes
        self.mlp_layer_regs = mlp_layer_regs
        let num_mlp_layers = self.mlp_layer_sizes.count

        precondition(self.mlp_layer_sizes[0]%2 == 0, 'u dummy, mlp_layer_sizes[0] % 2 != 0')
        precondition(self.mlp_layer_sizes.count == self.mlp_layer_regs.count, 'u dummy, layer_sizes != layer_regs!')

        //TODO: regularization
        var self.mf_user_embed = Embedding<Float>(vocabularySize: self.num_users, embeddingSize: self.mf_dim)
        var self.mf_item_embed = Embedding<Float>(vocabularySize: self.num_items, embeddingSize: self.mf_dim)
        var self.mlp_user_embed = Embedding<Float>(vocabularySize: self.num_users, embeddingSize: floor(self.mf_layer_sizes[0]/2))
        var self.mlp_item_embed = Embedding<Float>(vocabularySize: self.num_items, embeddingSize: floor(self.mf_layer_sizes[0]/2))

        //TODO: Extend it for n layers by using for loop
        //Currently only for 3 layers
        var dense1 = Dense(inputSize: self.mlp_layer_sizes[0], outputSize: self.mlp_layer_sizes[1], activation: relu)
        var dense2 = Dense(inputSize: self.mlp_layer_sizes[1], outputSize: self.mlp_layer_sizes[2], activation: relu)
        var dense3 = Dense(inputSize: self.mlp_layer_sizes[2], outputSize: self.mlp_layer_sizes[3], activation: relu)
        var final_dense = Dense(inputSize: (self.mlp_layer_sizes[3] + self.mf_dim), outputSize: 1, activation: sigmoid)

        @differentiable
        public func callAsFunction(_ user_indices: Int , _ item_indeces: Int) -> Tensor<Float>{
            var user_embed_mlp = self.mlp_user_embed(user_indices)
            var item_embed_mlp = self.mlp_item_embed(item_indices)
            var user_embed_mf = self.mf_user_embed(user_indices)
            var item_embed_mf = self.mf_item_embed(item_indices)

            var mf_vector = matmul(user_embed_mf,item_embed_mf)
            var mlp_vector = user_embed_mlp.concatenated(with:item_embed_mlp,alongAxis:-1)

            mlp_vector = mlp_vector.sequenced(through: dense1, dense2, dense3)
            var vector = mlp_vector.concatenated(with:mf_vector,alongAxis:-1)

            return vector.sequenced(through: final_dense)
        }
    }
}
