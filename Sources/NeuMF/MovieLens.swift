import Foundation
import TensorFlow
import Datasets

extension Sequence where Element : Collection {
    subscript(column column : Element.Index) -> [ Element.Iterator.Element ] {
        return map {$0[ column ]}
    }
}
extension Sequence where Iterator.Element: Hashable {
    func unique() -> [Iterator.Element]{
        var seen: Set<Iterator.Element> = []
        return filter{seen.insert($0).inserted}
    }
}

public struct MovieLens {

    public let trainUsers: [Float]
    public let testUsers: [Float]
    public let testData: [[Float]]
    public let items: [Float]
    public let numUsers: Int
    public let numItems: Int
    public let trainMatrix: [TensorPair<Int32,Float>]
    public let user2id: [Float:Int]
    public let id2user: [Int:Float]
    public let item2id: [Float:Int]
    public let id2item: [Int:Float]
    public let trainNegSampling: Tensor<Float>

    static func downloadMovieLensDatasetIfNotPresent() -> URL{
        let localURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let dataFolder = DatasetUtilities.downloadResource(
            filename: "ml-100k",
            fileExtension: "zip",
            remoteRoot: URL(string: "http://files.grouplens.org/datasets/movielens/")!,
            localStorageDirectory: localURL.appendingPathComponent("data/", isDirectory: true))

        return dataFolder}

    public init() {
        let trainFiles  = try! String(contentsOf: MovieLens.downloadMovieLensDatasetIfNotPresent().appendingPathComponent("u1.base"), encoding: .utf8)
        let testFiles = try! String(contentsOf: MovieLens.downloadMovieLensDatasetIfNotPresent().appendingPathComponent("u1.test"), encoding: .utf8)

        let trainData: [[Float]] = trainFiles.split(separator: "\n").map{ String($0).split(separator: "\t").compactMap{ Float(String($0))}}
        let testData: [[Float]] = testFiles.split(separator: "\n").map{ String($0).split(separator: "\t").compactMap{ Float(String($0))}}

        // let data = datad[0...5000]
        let trainUsers = trainData[column: 0].unique()
        let testUsers = testData[column: 0].unique()

        let items = trainData[column: 1].unique()

        let userIndex = 0...trainUsers.count-1
        let user2id = Dictionary(uniqueKeysWithValues: zip(trainUsers,userIndex))
        let id2user = Dictionary(uniqueKeysWithValues: zip(userIndex,trainUsers))

        let itemIndex = 0...items.count-1
        let item2id = Dictionary(uniqueKeysWithValues: zip(items,itemIndex))
        let id2item = Dictionary(uniqueKeysWithValues: zip(itemIndex,items))

        var trainNegSampling = Tensor<Float>(zeros: [trainUsers.count,items.count])

        var dataset:[TensorPair<Int32,Float>] = []

        for element in trainData{
            let uIndex = user2id[element[0]]!
            let iIndex = item2id[element[1]]!
            let rating = element[2]
            if (rating > 0){
              trainNegSampling[uIndex][iIndex] = Tensor(1.0)
            }
        }

        for element in trainData{
            let uIndex = user2id[element[0]]!
            let iIndex = item2id[element[1]]!
            let x = Tensor<Int32>([Int32(uIndex), Int32(iIndex)])
            dataset.append(TensorPair<Int32, Float>(first:x, second: [1]))

            for i in 0...3{
              var iIndex = Int.random(in:itemIndex)
              while(trainNegSampling[uIndex][iIndex].scalarized() == 1.0){
                iIndex = Int.random(in:itemIndex)
              }
              let x = Tensor<Int32>([Int32(uIndex), Int32(iIndex)])
              dataset.append(TensorPair<Int32, Float>(first: x, second: [0]))
            }
        }

        self.testData = testData
        self.numUsers = trainUsers.count
        self.numItems = items.count
        self.trainUsers = trainUsers
        self.testUsers = testUsers
        self.items = items
        self.user2id = user2id
        self.id2user = id2user
        self.item2id = item2id
        self.id2item = id2item
        self.trainMatrix = dataset
        self.trainNegSampling = trainNegSampling
    }
}
