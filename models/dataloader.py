# Import libraries 
from sklearn.model_selection import train_test_split


def split_train_val_and_test_set(X, Y, train_ratio, validation_ratio, test_ratio):
    """ Split datasets into training, validation, and test sets."""
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_ratio)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

class Dataset(torch.utils.data.Dataset):
  """ Characterizes a dataset for PyTorch."""
    
  def __init__(self, X, Y):
      """ Initialize the datasets."""
      self.X = X
      self.Y = Y

  def __len__(self):
      """ Return the number of training examples."""
      return self.X.shape[0]

  def __getitem__(self, index):
      """ Generate one sample of the training data."""
      x = self.X[index, :]
      y = self.Y[index, :]
      return x, y