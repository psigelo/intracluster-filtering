import torch
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
import tqdm

from core.data_selector import DataSelector


def one_hot_enc(cat_var):
    # Binary encoding
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = cat_var.reshape(len(cat_var), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


class CustomModel(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(CustomModel, self).__init__()
        self.cl1 = torch.nn.Linear(D_in, H1)
        self.cl2 = torch.nn.Linear(H1, H2)
        self.fc1 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        x = torch.sigmoid(self.cl1(x))
        x = torch.sigmoid(self.cl2(x))
        x = torch.sigmoid(self.fc1(x))
        return x

    def inspector_out(self, x):
        with torch.no_grad():
            x = torch.sigmoid(self.cl1(x))
            x = torch.sigmoid(self.cl2(x))
        return x


def iris_sample():
    total_iterations = 10000
    data_iris = load_iris()
    data = data_iris["data"]
    target = data_iris["target"]
    outs_posibilities = [0, 1, 2]
    labels = one_hot_enc(target)

    X_tr = Variable(torch.tensor(data, dtype=torch.float))
    y_tr = Variable(torch.tensor(labels, dtype=torch.float))

    data_selector = DataSelector(X_tr, y_tr, int(total_iterations * 0.7), 100)

    model2 = CustomModel(4, 5, 10, 3).to("cpu")
    loss_fn = torch.nn.MSELoss(reduction='sum')
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate, )

    for t in tqdm.tqdm(range(total_iterations), ascii=' >=', ncols=100):
        X_tr_filtered, y_tr_filtered = data_selector.get_train_data(epoch=t, model=model2, outs_posibilities=outs_posibilities)
        pred = model2(X_tr_filtered)
        loss = loss_fn(pred, y_tr_filtered)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    data_removed = data_selector.get_removed_data()
    print(data_removed)


if __name__ == "__main__":
    iris_sample()
