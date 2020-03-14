
import TensorFlow
import Foundation

let dataset = MovieLens()
let num_users = dataset.num_users
let num_items = dataset.num_items

let size:[Int] = [64, 32, 16, 8]
let regs:[Float] = [0.0, 0.0, 0.0, 0.0]
var model = NeuMF(num_users: num_users, num_items: num_items, mf_dim: 8, mf_reg: 0.0, mlp_layer_sizes: size, mlp_layer_regs: regs)

let optimizer = Adam(for: model, learningRate: 0.001)

withDevice(.gpu, 0) {
    for epoch in 1...3{
        var avg_loss: Float = 0.0
        Context.local.learningPhase = .training
        for (i,data) in dataset.user_item_rating.shuffled().enumerated(){
                let user_id = data[0]
                let item_id = data[1]
                let rating = Tensor(Float(data[2]))

                let input = Tensor<Int32>(shape: [2, 1], scalars: [Int32(user_id), Int32(item_id)])
                let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                let logits = model(input)
                return sigmoidCrossEntropy(logits: logits, labels:rating)
    //             return meanSquaredError(predicted: logits, expected: rating)
                                                            }
                optimizer.update(&model, along: grad)
                avg_loss = avg_loss + loss.scalarized()
                if(i+1)%200 == 0{
                    print(avg_loss/Float(i+1))
                }
        }
        print("Epoch: \(epoch)", "Current loss: \(avg_loss/(Float(dataset.user_item_rating.count)))")
    }
}
