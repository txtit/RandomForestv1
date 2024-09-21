import numpy as np
import pandas as pd
from collections import Counter

# Create dataset with the new feature "Biết bơi"
# Dữ liệu động vật
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
    'Biết bay': [
        0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0
    ],
    'Biết bơi': [
        0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 
        0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 
        0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 
        1, 0, 1, 0, 0, 1, 0, 0, 1, 0
    ]
}

# Nhãn: 0 = Thú, 1 = Cá, 2 = Chim
labels = [
    0, 1, 1, 1, 0, 0, 2, 0, 1, 2, 
    2, 0, 2, 2, 1, 2, 0, 1, 1, 2, 
    0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 
    1, 0, 2, 1, 0, 2, 1, 0, 1, 2
]
    

# Create DataFrame
animal_df = pd.DataFrame(data)
X = animal_df.values
y = np.array(labels)

# # Train-test split function
# def train_test_split(x, y, test_size=0.3, random_state=None):
#     if random_state is not None:
#         np.random.seed(random_state)
#     indices = np.arange(len(X))
# #      Kích thước của tập kiểm tra được tính bằng tỉ lệ phần trăm test_size.
# #      Đảm bảo rằng mỗi chỉ số chỉ được chọn một lần (không thay thế).
#     test_indices = np.random.choice(indices, size=int(len(X) * test_size), replace=False)
# #  phan con lai
#     train_indices = np.setdiff1d(indices, test_indices)
# #  tach du lieu
#     X_train, X_test = X[train_indices], X[test_indices]
# #  tach nhan
#     y_train, y_test = y[train_indices], y[test_indices]
#     return X_train, X_test, y_train, y_test

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Bootstrap sampling function
def bootstrap_sample(X, y):
    n_samples = len(X)
    indices = np.random.choice(n_samples, n_samples, replace=True)  # Lấy mẫu có hoàn lại
    return X[indices], y[indices]

# Tạo tập huấn luyện
X_train, y_train = bootstrap_sample(X, y)

# Tạo tập kiểm tra từ các mẫu không có trong tập huấn luyện
def create_test_set(X, y, X_train):
    test_indices = np.setdiff1d(np.arange(len(X)), np.unique(X_train, axis=0, return_index=True)[1])
    return X[test_indices], y[test_indices]

X_test, y_test = create_test_set(X, y, X_train)

# Decision Tree Class
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    # Training method
    def fit(self, X, y, depth=0):
        #  kiem tra nhan giong nhau
        if len(set(y)) == 1:
            return y[0]
        if self.max_depth is not None and depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0] # tra ve nhan pho bien nhat

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

    def _information_gain(self, X, y, feature):
        entropy_before = self._entropy(y)
        values, counts = np.unique(X[:, feature], return_counts=True)
        # Tỷ số này tính tỉ lệ của các mẫu có giá trị values[i] so với toàn bộ tập dữ liệu. 
        # Nó thể hiện phần trăm mẫu có giá trị này trong tổng số mẫu
        weighted_entropy = sum((counts[i] / len(y)) * self._entropy(y[X[:, feature] == values[i]])
                               for i in range(len(values)))
        return entropy_before - weighted_entropy
    
    
# Hàm này tính toán mức độ không chắc chắn (hay độ hỗn loạn) trong tập nhãn y.
# Entropy càng cao có nghĩa là các nhãn phân bố đều hơn,
# trong khi entropy thấp có nghĩa là một nhãn nào đó chiếm ưu thế hơn các nhãn khác.
    def _entropy(self, y):
        proportions = [count / len(y) for count in Counter(y).values()]
        return -sum(p * np.log2(p) for p in proportions if p > 0)

# Hàm predict sử dụng cây quyết định đã được huấn luyện để dự đoán nhãn cho mỗi mẫu trong
# tập dữ liệu đầu vào. Nó trả về một danh sách chứa các nhãn dự đoán cho từng mẫu,
# giúp bạn đánh giá mô hình trên dữ liệu mới.
    def predict(self, X):
        return [self._predict_sample(x, self.tree) for x in X]
#  du doan du lieu moi
    def _predict_sample(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        feature = list(tree.keys())[0]
        value = x[feature]
        subtree = tree[feature].get(value, Counter([0]).most_common(1)[0][0])
        return self._predict_sample(x, subtree)

# Random Forest Class
class RandomForest:
    def __init__(self, n_estimators=300, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(len(X), len(X), replace=True)
            sample_X = X[sample_indices]
            sample_y = y[sample_indices]
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return [Counter(row).most_common(1)[0][0] for row in tree_preds.T]

# tao va huan luyen rf
rf = RandomForest(n_estimators=300, max_depth=20)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Accuracy score function
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")

# Sample prediction
sample = np.array([[0.2, 7, 0, 1]])  # Example new data
prediction = rf.predict(sample)
animal_types = ["Thú", "Cá", "Chim"]
print(f"Dự đoán nhãn cho mẫu mới: {animal_types[prediction[0]]}")
