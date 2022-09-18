import numpy as np
from sklearn.decomposition import TruncatedSVD
from sub_models import *


class SVD():
    def __init__(self, sparse_matrix, truncate=100, seed=42):
        super(SVD, self).__init__()
        self.sparse_matrix = sparse_matrix
        self.truncate = truncate
        # self.model = TruncatedSVD(n_components=self.truncate, random_state=seed)

    def train(self):
        # self.model.fit(self.sparse_matrix)
        self.matrixs = np.linalg.svd(np.array(self.sparse_matrix), full_matrices=False)

    def predict(self):
        u, s, vh = self.matrixs
        truncated_u = u[:,:self.truncate]
        truncated_s = s[:self.truncate]
        truncated_vh = vh[:self.truncate,:]
        restore_matrix = np.dot(truncated_u, np.dot(np.diag(truncated_s), truncated_vh)).round().astype(int)

        return restore_matrix


class MatrixFactorization:
    def __init__(self, R: np.ndarray, k: int, lr: float, regularization: float, epochs: int, verbose: bool =False) -> None:
        """
        :param R: rating matrix
        :param k: latent parameter
        :param lr: learning rate
        :param regularization: regularization term for update
        :param epochs: training epochs
        :param verbose: print status
        """

        self._R = R
        self._n_users, self._n_items = R.shape
        self._k = k
        self._lr = lr
        self._regularization = regularization
        self._epochs = epochs
        self._verbose = verbose


    def train(self) -> None:

        # latent features
        self._P = np.random.normal(size=(self._n_users, self._k))
        self._Q = np.random.normal(size=(self._n_items, self._k))

        # biases
        self._bu = np.zeros(self._n_users)
        self._bi = np.zeros(self._n_items)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        # train while epochs
        self._training_process = []
        for epoch in range(self._epochs):

            # rating이 0이 아닌 index로 train
            for i in range(self._n_users):
                for j in range(self._n_items):
                    if self._R[i, j] > 0:
                        self.gradient_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))


    def cost(self) -> None:
        """
        compute root mean square error
        :return: rmse cost
        """

        # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
        xi, yi = self._R.nonzero()
        predicted = self.predict()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - predicted[x, y], 2)
        return np.sqrt(cost / len(xi))


    def gradient(self, error: float, i: int, j: int) -> tuple:
        """
        gradient of latent feature for GD

        :param error: rating - prediction error
        :param i: user index
        :param j: item index
        :return: gradient of latent feature tuple
        """

        dp = (error * self._Q[j, :]) - (self._regularization * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._regularization * self._Q[j, :])
        return dp, dq


    def gradient_descent(self, i: int, j: int, rating: int) -> None:
        """
        graident descent function

        :param i: user index of matrix
        :param j: item index of matrix
        :param rating: rating of (i,j)
        """

        # get error
        prediction = self.predict_train(i, j)
        error = rating - prediction

        # update biases
        self._bu[i] += self._lr * (error - self._regularization * self._bu[i])
        self._bi[j] += self._lr * (error - self._regularization * self._bi[j])

        # update latent feature
        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._lr * dp
        self._Q[j, :] += self._lr * dq


    def predict_train(self, i: int, j: int) -> float:
        """
        get predicted rating: user_i, item_j
        :return: prediction of r_ij
        """
        return self._b + self._bu[i] + self._bi[j] + self._P[i, :].dot(self._Q[j, :].T)


    def predict(self) -> np.ndarray:
        """
        computer complete matrix PXQ + P.bias + Q.bias + global bias

        - PXQ 행렬에 _bu[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
        - _bi[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
        - b를 더하는 것은 각 element마다 bias를 더해주는 것

        - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을 해주기위해 차원을 추가하는 것.

        :return: complete matrix R^
        """
        return self._b + self._bu[:, np.newaxis] + self._bi[np.newaxis:, ] + self._P.dot(self._Q.T)


class AlternatingLeastSquares:
    def __init__(self, R: np.ndarray, k: int, regularization: float, epochs: int, verbose: bool =False) -> None:
        """
        :param R: rating matrix
        :param k: latent parameter
        :param regularization: regularization term for update
        :param epochs: training epochs
        :param verbose: print status
        """
        self._R = R
        self._n_users, self._n_items = R.shape
        self._k = k
        self._regularization = regularization
        self._epochs = epochs
        self._verbose = verbose


    def train(self) -> None:
        # init latent features
        self._users = np.random.normal(size=(self._n_users, self._k))
        self._items = np.random.normal(size=(self._n_items, self._k))

        # train while epochs
        self._training_process = []
        self._user_error = 0; self._item_error = 0;
        for epoch in range(self._epochs):
            for i, Ri in enumerate(self._R):
                self._users[i] = self.user_latent(i, Ri)
                self._user_error = self.cost()

            for j, Rj in enumerate(self._R.T):
                self._items[j] = self.item_latent(j, Rj)
                self._item_error = self.cost()

            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))


    def cost(self) -> float:
        """
        compute root mean square error
        :return: rmse cost
        """
        xi, yi = self._R.nonzero()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - self.predict_train(x, y), 2)
        return np.sqrt(cost/len(xi))


    def user_latent(self, i: int, Ri: np.ndarray) -> np.ndarray:
        """
        :param i: user index
        :param Ri: Rating of user index i
        :return: convergence value of user latent of i index
        """

        du = np.linalg.solve(np.dot(self._items.T, np.dot(np.diag(Ri), self._items)) +
                             self._regularization * np.eye(self._k),
                             np.dot(self._items.T, np.dot(np.diag(Ri), self._R[i].T))).T
        return du

    def item_latent(self, j: int, Rj: np.ndarray) -> np.ndarray:
        """
        :param j: item index
        :param Rj: Rating of item index j
        :return: convergence value of itemr latent of j index
        """

        di = np.linalg.solve(np.dot(self._users.T, np.dot(np.diag(Rj), self._users)) +
                             self._regularization * np.eye(self._k),
                             np.dot(self._users.T, np.dot(np.diag(Rj), self._R[:, j])))
        return di


    def predict_train(self, i: int, j: int) -> float:
        """
        get predicted rating: user_i, item_j
        :return: prediction of r_ij
        """
        return self._users[i, :].dot(self._items[j, :].T)


    def predict(self) -> np.ndarray:
        """
        :return: complete matrix R^
        """
        return self._users.dot(self._items.T)


class FactorizationMachineModel:

    def __init__(self, data_X, data_y, criterion, field_dims: np.ndarray, embed_dim: int,
                batch_size: int, data_shuffle: bool, epochs: int, learning_rate: float, weight_decay: float):
        super().__init__()

        self.criterion = criterion
        self.field_dims = field_dims
        self.batch_size = batch_size
        self.data_shuffle = data_shuffle
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gpu_idx = 0
        self.log_interval = 100

        self.dataloader = DataLoader(TensorDataset(torch.LongTensor(np.array(data_X)), torch.IntTensor(np.array(data_y))), batch_size=self.batch_size, shuffle=self.data_shuffle)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(self.gpu_idx) if torch.cuda.is_available() else "cpu")

        self.model = _FactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, data_loader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0):
                fields = fields.to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class FieldAwareFactorizationMachineModel:

    def __init__(self, data_X, data_y, criterion, field_dims: np.ndarray,
                batch_size: int, data_shuffle: bool, embed_dim: int, epochs: int, learning_rate: float, weight_decay: float):
        super().__init__()

        self.criterion = criterion
        self.field_dims = field_dims
        self.batch_size = batch_size
        self.data_shuffle = data_shuffle
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gpu_idx = 0
        self.log_interval = 100

        self.dataloader = DataLoader(TensorDataset(torch.LongTensor(np.array(data_X)), torch.IntTensor(np.array(data_y))), batch_size=self.batch_size, shuffle=self.data_shuffle)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(self.gpu_idx) if torch.cuda.is_available() else "cpu")

        self.model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, data_loader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0):
                fields = fields.to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts



class NeuralCollaborativeFiltering:

    def __init__(self, data_X, data_y, criterion, field_dims: np.ndarray, embed_dim: int,
                batch_size: int, data_shuffle: bool, epochs: int, learning_rate: float, weight_decay: float):
        super().__init__()

        self.criterion = criterion
        self.field_dims = field_dims
        self.batch_size = batch_size
        self.data_shuffle = data_shuffle
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gpu_idx = 0
        self.log_interval = 100

        self.dataloader = DataLoader(TensorDataset(torch.LongTensor(np.array(data_X)), torch.IntTensor(np.array(data_y))), batch_size=self.batch_size, shuffle=self.data_shuffle)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(self.gpu_idx) if torch.cuda.is_available() else "cpu")

        self.model = _NeuralCollaborativeFiltering(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, data_loader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0):
                fields = fields.to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class WideAndDeepModel:

    def __init__(self, data_X, data_y, criterion, field_dims: np.ndarray, embed_dim: int,
                batch_size: int, data_shuffle: bool, epochs: int, learning_rate: float, weight_decay: float):
        super().__init__()

        self.criterion = criterion
        self.field_dims = field_dims
        self.batch_size = batch_size
        self.data_shuffle = data_shuffle
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gpu_idx = 0
        self.log_interval = 100

        self.dataloader = DataLoader(TensorDataset(torch.LongTensor(np.array(data_X)), torch.IntTensor(np.array(data_y))), batch_size=self.batch_size, shuffle=self.data_shuffle)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(self.gpu_idx) if torch.cuda.is_available() else "cpu")

        self.model = _WideAndDeepModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, data_loader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0):
                fields = fields.to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class DeepCrossNetworkModel:

    def __init__(self, data_X, data_y, criterion, field_dims: np.ndarray, embed_dim: int,
                batch_size: int, data_shuffle: bool, epochs: int, learning_rate: float, weight_decay: float):
        super().__init__()

        self.criterion = criterion
        self.field_dims = field_dims
        self.batch_size = batch_size
        self.data_shuffle = data_shuffle
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gpu_idx = 0
        self.log_interval = 100

        self.dataloader = DataLoader(TensorDataset(torch.LongTensor(np.array(data_X)), torch.IntTensor(np.array(data_y))), batch_size=self.batch_size, shuffle=self.data_shuffle)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(self.gpu_idx) if torch.cuda.is_available() else "cpu")

        self.model = _DeepCrossNetworkModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, data_loader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.dataloader, smoothing=0, mininterval=1.0):
                fields = fields.to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts
