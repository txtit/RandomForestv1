import numpy as np
import pandas as pd
from collections import Counter

# tao du lieu

# Dữ liệu động vật mở rộng
# Dữ liệu động vật mở rộng với độ dài đồng nhất (40 phần tử)
data = {
    'Cân nặng (kg)': [
        220, 150000, 30, 200, 85, 5000, 50, 120, 10000, 3, 
        0.02, 300, 0.5, 1.2, 0.1, 1.5, 0.3, 0.8, 1.0, 0.2, 
        500, 80, 60, 75, 1.5, 0.4, 6000, 0.5, 0.3, 0.1, 
        250, 0.6, 0.7, 0.2, 2.0, 1.8, 3.0, 1.2, 2.5, 0.7
    ],
    'Tuổi thọ trung bình (năm)': [
        15, 70, 20, 80, 23, 40, 12, 8, 60, 4, 
        1, 30, 10, 15, 5, 8, 6, 4, 3, 7, 
        5, 10, 7, 12, 8, 4, 6, 14, 3, 9, 
        2, 5, 7, 6, 11, 8, 3, 5, 6, 10
    ],
    'Tốc độ chạy (km/h)': [
        80, 10, 6, 1, 50, 40, 70, 90, 0, 12, 
        2, 100, 30, 25, 20, 15, 5, 8, 25, 10,
        80, 70, 60, 50, 40, 30, 20, 10, 5, 2, 
        100, 90, 80, 70, 60, 50, 40, 30, 20, 10
    ],
    'Loại thức ăn': [
        0, 1, 1, 2, 2, 0, 1, 0, 2, 1, 
        2, 0, 2, 2, 1, 2, 0, 1, 1, 2, 
        0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 
        1, 0, 2, 1, 0, 2, 1, 0, 1, 2
    ]
}

labels = [
    0, 1, 2, 1, 0, 0, 2, 0, 1, 2, 
    2, 0, 2, 2, 1, 2, 0, 1, 1, 2, 
    0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 
    1, 0, 2, 1, 0, 2, 1, 0, 1, 2
]  # Các nhãn: 0 = Thú, 1 = Cá, 2 = Chim
# dataframe
animal_df = pd.DataFrame(data)
X = animal_df.values
y = np.array(labels)

# chia du lieu thanh train & test
def train_test_split(x, y, test_size=0.3, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    test_indices = np.random.choice(indices, size=int(len(X) * test_size), replace=False)
    train_indices = np.setdiff1d(indices , test_indices)
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Decision Tree
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None


# Training DT


    def fit(self, X, y, depth=0):
        # neu tat ca cac label giong nhau
        if len(set(y)) == 1:
            return y[0]
        # Neu da dat den do sau toi da
        if self.max_depth is not None and depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]

        best_feature = self._best_feature(X, y)
        if best_feature is None:
            return Counter(y).most_common(1)[0][0]

        tree = {best_feature: {}}
        for value in set(X[:, best_feature]):
            sub_X = X[X[:, best_feature] == value]
            sub_y = y[X[:, best_feature] == value]
            tree[best_feature][value] = self.fit(sub_X, sub_y, depth + 1)
        self.tree = tree 
        return self.tree


    def _best_feature(self, X, y):
        best_feature = None
        best_gain = 0
        num_features = X.shape[1]

        for feature in range(num_features):
            gain = self._information_gain(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return best_feature


    # Count gain
    def _information_gain(self, X, y, feature):
        # Entropy của nhãn trước khi phân chia.
        entropy_before = self._entropy(y)
        values, counts = np.unique(X[:, feature], return_counts=True)
        # Entropy sau khi phân chia, trọng số theo số lượng mẫu.
        weighted_entropy = sum((counts[i] / len(y)) * self._entropy(y[X[:, feature] == values[i]])
            for i in range(len(values)))
        # Lợi ích thông tin là sự khác biệt giữa entropy trước và sau phân chia.
        return entropy_before - weighted_entropy


    def _entropy(self, y):
        # Tỉ lệ của mỗi nhãn trong tập dữ liệu
        proportions = [count / len(y) for count in Counter(y).values()]
        # Công thức tính entropy.
        return -sum(p * np.log2(p) for p in proportions if p > 0)


    # Dự đoán nhãn cho các mẫu dữ liệu mới
    def predict(self, X):
        return [self._predict_sample(x, self.tree) for x in X]


    # Dự đoán cho 1 mẫu
    def _predict_sample(self, x, tree):
        # Nếu nút hiện tại là nhãn (không phải từ điển), trả về nhãn.
        if not isinstance(tree, dict):
            return tree
        # Thuộc tính phân chia ở nút hiện tại.
        feature = list(tree.keys())[0]
        # : Giá trị của thuộc tính cho mẫu.
        value = x[feature]
        # Lấy nhánh con hoặc trả về nhãn phổ biến nhất nếu không tìm thấy nhánh con.
        subtree = tree[feature].get(value, Counter([0]).most_common(1)[0][0])
        # Đệ quy dự đoán nhãn cho mẫu trong nhánh con.
        return self._predict_sample(x, subtree)


# khoi tao RF
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        # Số lượng cây quyết định trong rừng. Mặc định là 100.
        self.n_estimators = n_estimators
        # Độ sâu tối đa của mỗi cây quyết định. Nếu không có,
        # cây có thể phát triển đến khi dữ liệu không còn phân loại được.
        self.max_depth = max_depth
        # Danh sách để lưu trữ các cây quyết định trong rừng.
        self.trees = []

    # Huấn Luyện Random Forest
    def fit(self, X, y):
    # Lặp qua số lượng cây quyết định (estimators) để huấn luyện.
        for _ in range(self.n_estimators):
        # Tạo mẫu bootstrap (lặp lại) từ chỉ số của dữ liệu. replace=True cho phép lấy lại mẫu.
            sample_indices = np.random.choice(len(X), len(X), replace=True)
        # Chuyen doi thanh so nguyen
            sample_indices = sample_indices.astype(int)
        #    Lấy dữ liệu và nhãn cho mẫu bootstrap.
            sample_X = X[sample_indices]
            sample_y = y[sample_indices]
        # Tạo một cây quyết định mới.
            tree = DecisionTree(max_depth=self.max_depth)
        # Huấn luyện cây quyết định với mẫu bootstrap.
            tree.fit(sample_X, sample_y)
        # Thêm cây vào danh sách cây trong rừng.
            self.trees.append(tree)


    def predict(self, X):
    #  Dự đoán của tất cả các cây quyết định cho dữ liệu X. tree_preds
    # là một mảng 2D, trong đó mỗi hàng chứa dự đoán của một cây.
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
    # Đối với mỗi mẫu dữ liệu, lấy dự đoán phổ biến nhất từ các cây.
    # Counter(row).most_common(1)[0][0] trả về nhãn phổ biến nhất cho mỗi mẫu dự đoán.
        return [Counter(row).most_common(1)[0][0] for row in tree_preds.T]


rf = RandomForest(n_estimators=100, max_depth=5)
rf.fit(X_train, y_train)

# rf = RandomForest(n_estimators=10, max_depth=3): Tạo một đối tượng RandomForest
# với 10 cây quyết định (n_estimators=10) và độ sâu tối đa cho mỗi cây là 3 (max_depth=3).

# rf.fit(X_train, y_train): Huấn luyện mô hình Random Forest với
# tập dữ liệu huấn luyện X_train và nhãn y_train. Đây là bước tạo ra các cây quyết định dựa trên dữ liệu huấn luyện.


y_pred = rf.predict(X_test)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")


# y_pred = rf.predict(X_test): Dự đoán nhãn cho dữ liệu kiểm tra X_test bằng mô hình Random Forest đã huấn luyện.
# def accuracy_score(y_true, y_pred): Định nghĩa hàm tính độ chính xác của mô hình. Hàm này so sánh các nhãn thực tế (y_true) với các nhãn dự đoán (y_pred), và tính tỷ lệ chính xác.
# accuracy = accuracy_score(y_test, y_pred): Tính độ chính xác của mô hình trên tập kiểm tra.
# print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%"): In ra độ chính xác của mô hình dưới dạng phần trăm.


sample = np.array([[0.5, 5, 40, 1]])  # Ví dụ một dữ liệu mới

prediction = rf.predict(sample)
animal_types = ["Thú", "Cá", "Chim"]  # Các nhãn tương ứng
print(f"Dự đoán nhãn cho mẫu mới: {animal_types[prediction[0]]}")


# sample = np.array([[250, 20, 60, 0]]): Tạo một ví dụ dữ liệu mới với các đặc trưng. Ở đây, dữ liệu mẫu bao gồm các giá trị cho các đặc trưng như cân nặng, tuổi thọ, tốc độ chạy, và loại thức ăn.
# prediction = rf.predict(sample): Dự đoán nhãn cho dữ liệu mẫu bằng mô hình Random Forest.
# animal_types = ['Thú', 'Cá', 'Chim']: Định nghĩa các nhãn tương ứng với các lớp trong dữ liệu. Ví dụ: 0 là 'Thú', 1 là 'Cá', và 2 là 'Chim'.
# print(f"Dự đoán nhãn cho mẫu mới: {animal_types[prediction[0]]}"): In ra nhãn dự đoán cho dữ liệu mới.
