
import TensorFlow
import Foundation
import Batcher

let dataset = MovieLens()
let numUsers = dataset.numUsers
let numItems = dataset.numItems

let batcher = Batcher(on: dataset.trainMatrix, batchSize: 1024, shuffle: true)

print("Number of datapoints", dataset.trainMatrix.count)
print("Number of users", numUsers)
print("Number of items", numItems)
// print(_ExecutionContext.global.deviceNames)

let size:[Int] = [64, 32, 16, 8]
let regs:[Float] = [0.0, 0.0, 0.0, 0.0]
var model = NeuMF(numUsers: numUsers, numItems: numItems, mfDim: 8, mfReg: 0.0, mlpLayerSizes: size, mlpLayerRegs: regs)

let optimizer = Adam(for: model, learningRate: 0.001)

for epoch in 1...2{
    var avgLoss: Float = 0.0
    Context.local.learningPhase = .training
    for data in batcher.sequenced(){
            let userId = data.first
            let rating = data.second
            let (loss, grad) = valueWithGradient(at: model){model -> Tensor<Float> in
            let logits = model(userId)
            return sigmoidCrossEntropy(logits: logits, labels: rating)}

            optimizer.update(&model, along: grad)
            avgLoss = avgLoss + loss.scalarized()
    }
    print("Epoch: \(epoch)", "Current loss: \(avgLoss)")
}


for user in dataset.testUsers{
    var negativeItem: [Float] = []
    var output: [Float] = []
    for item in dataset.items{
        let userIndex = dataset.user2id[user]!
        let itemIndex = dataset.item2id[item]!

        if dataset.trainNegSampling[userIndex][itemIndex].scalarized() == 0{
            let input =  Tensor<Int32>(shape: [1, 2],scalars: [Int32(userIndex),Int32(itemIndex)])
            output.append(model(input).scalarized())
            negativeItem.append(item)
        }
    }
    let itemScore = Dictionary(uniqueKeysWithValues: zip(negativeItem,output))
    let sortedItemScore = itemScore.sorted {$0.1 > $1.1}
    let topK = sortedItemScore.prefix(10)

    print("User:", user , terminator:"\t")
    print("Top 10 Recommended Items:", terminator:"\t")
    for (key,_) in topK{
        print(key, terminator: "\t")
    }
    print(terminator: "\n")
}
