import Foundation
import TensorFlow
import Datasets

extension Sequence where Element : Collection {
    subscript(column column : Element.Index) -> [ Element.Iterator.Element ] {
        return map { $0[ column ] }
    }
}
extension Sequence where Iterator.Element: Hashable {
    func unique() -> [Iterator.Element] {
        var seen: Set<Iterator.Element> = []
        return filter { seen.insert($0).inserted }
    }
}

public struct MovieLens {

    public let users: [Float]
    public let items: [Float]
    public let num_users: Int
    public let num_items: Int
    public let user_item_rating: [[Int]]
    public let rating: [Float]
    public let user2id: [Float:Int]
    public let id2user: [Int:Float]
    public let item2id: [Float:Int]
    public let id2item: [Int:Float]
    public let neg_sampling: Tensor<Float>

    static func downloadMovieLensDatasetIfNotPresent() -> String{
        let localURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let dataFolder = DatasetUtilities.downloadResource(
            filename: "ml-100k",
            fileExtension: "zip",
            remoteRoot: URL(string: "http://files.grouplens.org/datasets/movielens/")!,
            localStorageDirectory: localURL.appendingPathComponent("data/", isDirectory: true))

        return try! String(contentsOf: dataFolder.appendingPathComponent("u1.base"), encoding: .utf8)}

    public init() {
        let dataFiles  = MovieLens.downloadMovieLensDatasetIfNotPresent()
        let data: [[Float]] = dataFiles.split(separator: "\n").map{ String($0).split(separator: "\t").compactMap{ Float(String($0)) } }

        // let data = dataRecords[0...30000]
        let users = data[column: 0].unique()
        let items = data[column: 1].unique()
        let rating = data[column: 2]

        let user_index = 0...users.count-1
        let user2id:[Float:Int] = Dictionary(uniqueKeysWithValues: zip(users,user_index))
        let id2user:[Int:Float] = Dictionary(uniqueKeysWithValues: zip(user_index,users))

        let item_index = 0...items.count-1
        let item2id:[Float:Int] = Dictionary(uniqueKeysWithValues: zip(items,item_index))
        let id2item:[Int:Float] = Dictionary(uniqueKeysWithValues: zip(item_index,items))

        var neg_sampling = Tensor<Float>(zeros: [users.count,items.count])

        var dataset:[[Int]] = []
        for element in data{
            let u_index = user2id[element[0]]!
            let i_index = item2id[element[1]]!
            let rating = element[2]
            if (rating > 0){
              dataset.append([u_index,i_index, 1])
              neg_sampling[u_index][i_index] = Tensor(1.0)
            }
        }

        for u_index in user_index{
            for i in 0...4{
              var i_index = Int.random(in:item_index)
              while(neg_sampling[u_index][i_index].scalarized() == 1.0){
                i_index = Int.random(in:item_index)
              }
              dataset.append([u_index,i_index, 0])
            }
        }

        self.num_users = users.count
        self.num_items = items.count
        self.users = users
        self.items = items
        self.rating = rating
        self.user2id = user2id
        self.id2user = id2user
        self.item2id = item2id
        self.id2item = id2item
        self.user_item_rating = dataset
        self.neg_sampling = neg_sampling
    }
}
