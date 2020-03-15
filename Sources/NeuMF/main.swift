
import TensorFlow
import Foundation
import Batcher

let dataset = MovieLens()
let num_users = dataset.num_users
let num_items = dataset.num_items

let batcher = Batcher(on: dataset.user_item_rating, batchSize: 1024, shuffle: true)

print("Number of datapoints", dataset.user_item_rating.count)
print("Number of users", num_users)
print("Number of items", num_items)
// print(_ExecutionContext.global.deviceNames)

let size:[Int] = [64, 32, 16, 8]
let regs:[Float] = [0.0, 0.0, 0.0, 0.0]
var model = NeuMF(num_users: num_users, num_items: num_items, mf_dim: 8, mf_reg: 0.0, mlp_layer_sizes: size, mlp_layer_regs: regs)

let optimizer = Adam(for: model, learningRate: 0.001)

for epoch in 1...50{
    var avg_loss: Float = 0.0
    Context.local.learningPhase = .training
    for data in batcher.sequenced(){
            let user_id = data.first
            let rating = data.second
            let (loss, grad) = valueWithGradient(at: model){model -> Tensor<Float> in
            let logits = model(user_id)
            return sigmoidCrossEntropy(logits: logits, labels: rating)}

            optimizer.update(&model, along: grad)
            avg_loss = avg_loss + loss.scalarized()
    }
    print("Epoch: \(epoch)", "Current loss: \(avg_loss)")
}
