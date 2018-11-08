import os, timeit
import numpy as np
from torch import nn, Tensor, optim, cuda, device
from torch.multiprocessing import cpu_count
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, auc

class Torch_model():
    """
    """

    def __init__(self, fold_dir, fold_name, model = None, n_channels = 4, rf_dim = 1, n_epochs = 100):
        super(Torch_model, self).__init__()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters())
        self.n_channels = n_channels
        self.rf_width = 2 * np.max(rf_dim) + 1
        self.n_epochs = n_epochs

        self.X_train = None
        self.y_train = None
        self.train_index = 0
        self.X_test = None
        self.y_test = None
        self.test_index = 0

        self.fold_dir = fold_dir
        self.fold_name = fold_name

        self.device = 'cpu'
        # if cuda.is_available():
        #     print ('Using GPU')
        #     self.device = device('cuda')
        # self.device = 'cuda:0'

    @staticmethod
    def hello_world():
        print('NN Model')

    @staticmethod
    def get_settings():
        return {}

    def initialise_train_data(self, n_datapoints, data_dimensions):
        self.X_train = np.empty([np.sum(n_datapoints), data_dimensions])
        self.y_train = np.empty(np.sum(n_datapoints))
        self.train_index = 0

    def add_train_data(self, batch_X_train, batch_y_train):
        """
        Add a batch of training data to the whole training data pool

        Args:
            batch_X_train: batch of training data
            batch_y_train: batch of training labels
        """
        self.X_train[self.train_index : self.train_index + batch_X_train.shape[0], :] = batch_X_train
        self.y_train[self.train_index : self.train_index + batch_y_train.shape[0]] = batch_y_train
        self.train_index += batch_X_train.shape[0]

    def initialise_test_data(self, n_datapoints, data_dimensions):
        self.X_test = np.empty([np.sum(n_datapoints), data_dimensions])
        self.y_test = np.empty(np.sum(n_datapoints))
        self.test_index = 0

    def add_test_data(self, batch_X_test, batch_y_test):
        """
        Add a batch of testing data to the whole testing data pool
        All testing data is saved in a svmlight file
        """
        self.X_test[self.test_index : self.test_index + batch_X_test.shape[0], :] = batch_X_test
        self.y_test[self.test_index : self.test_index + batch_y_test.shape[0]] = batch_y_test
        self.test_index += batch_X_test.shape[0]

    def forward(self, model, dl, optimizer=None):
        total_acc = 0
        total_loss = 0
        # print('éjhjh', dl.shape)
        roc_aucs = []
        c = 0
        criterion = nn.BCEWithLogitsLoss()
        for inputs, labels in dl:
            c += 1
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            probas_ = model(inputs).squeeze()
            threshold = 0.5
            acc = ((probas_ > threshold) == (labels == 1)).float().mean().item()
            fpr, tpr, thresholds = roc_curve(labels.detach().numpy(), probas_.detach().numpy())
            roc_aucs.append(auc(fpr, tpr))
            loss = criterion(probas_, labels.float())
            total_loss += loss.item()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return np.nanmean(roc_aucs), total_acc / c, total_loss / c

    def train(self):
        self.model.to(self.device)
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
        ds_train = TensorDataset(
            Tensor(self.X_train.reshape(-1, self.n_channels, self.rf_width, self.rf_width, self.rf_width)),
            Tensor(self.y_train)
            )
        ds_test = TensorDataset(
            Tensor(self.X_test.reshape(-1, self.n_channels, self.rf_width, self.rf_width, self.rf_width)),
            Tensor(self.y_test)
            )
        dl_train = DataLoader(ds_train, batch_size=128, num_workers=cpu_count(), pin_memory=True)
        dl_test = DataLoader(ds_test, batch_size=1024, num_workers=cpu_count(), pin_memory=True)
        for e in range(self.n_epochs):
            a = timeit.default_timer()
            train_roc_auc, train_acc, train_loss = self.forward(self.model, dl_train, self.optimizer)
            test_roc_auc, test_acc, test_loss = self.forward(self.model, dl_test)
            print(e, train_roc_auc, test_roc_auc, train_acc, train_loss, test_acc, test_loss)
            print('this took' + str(timeit.default_timer() - a))
        # try:
        #     for e in range(1000):
        #         train_acc, train_loss = self.forward(self.model, dl_train, self.optimizer)
        #         test_acc, test_loss = self.forward(self.model, dl_test)
        #         print(train_acc, train_loss, test_acc, test_loss)
        # except:
        #     pass
        train_eval = []
        return self.model, train_eval

    def predict(self, data):
        n_samples = data.shape[0]
        data = Tensor(data.reshape(n_samples, self.n_channels, self.rf_width, self.rf_width, self.rf_width))
        print('joooooooooioioioioioio',data.shape)
        probas_ = self.model(data)
        probas_ = probas_.data.numpy().squeeze()
        print('pppppppppopopopopo', probas_.shape)
        return probas_

    def predict_test_data(self):
        probas_ = self.predict(self.X_test)
        return probas_

    def get_test_labels(self):
        return self.y_test
