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

    public let user_pool: [Float]
    public let item_pool: [Float]
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
            filename: "ml-1m",
            fileExtension: "zip",
            remoteRoot: URL(string: "http://files.grouplens.org/datasets/movielens/")!,
            localStorageDirectory: localURL.appendingPathComponent("data/", isDirectory: true))

        return try! String(contentsOf: dataFolder.appendingPathComponent("ratings.dat"), encoding: .utf8)
    }

    public init() {
        let data  = MovieLens.downloadMovieLensDatasetIfNotPresent()
        let df_data: [[Float]] = data.split(separator: "\n").map{ String($0).split(separator: ":").compactMap{ Float(String($0)) } }

        let df = df_data[0..<30000]
        let user_pool = df[column: 0].unique()
        let item_pool = df[column: 1].unique()
        let rating = df[column: 2]

        let user_index = 0...user_pool.count-1
        let user2id:[Float:Int] = Dictionary(uniqueKeysWithValues: zip(user_pool,user_index))
        let id2user:[Int:Float] = Dictionary(uniqueKeysWithValues: zip(user_index,user_pool))

        let item_index = 0...item_pool.count-1
        let item2id:[Float:Int] = Dictionary(uniqueKeysWithValues: zip(item_pool,item_index))
        let id2item:[Int:Float] = Dictionary(uniqueKeysWithValues: zip(item_index,item_pool))

        var neg_sampling = Tensor<Float>(zeros: [user_pool.count,item_pool.count])
        for element in df{
            let u_index = user2id[element[0]]!
            let i_index = item2id[element[1]]!
            neg_sampling[u_index][i_index] = Tensor(1.0)
        }
        var dataset:[[Int]] = []
        for user_id in user_index{
            for item_id in item_index{
                let rating  = Int(neg_sampling[user_id][item_id].scalarized())
                dataset.append([user_id,item_id, rating])
            }
        }
        self.num_users = user_pool.count
        self.num_items = item_pool.count
        self.user_pool = user_pool
        self.item_pool = item_pool
        self.rating = rating
        self.user2id = user2id
        self.id2user = id2user
        self.item2id = item2id
        self.id2item = id2item
        self.user_item_rating = dataset
        self.neg_sampling = neg_sampling
    }
}
