import sn
from src.data.load_data import Data

num_classes_for_Food101: int = 101
data = Data()
train_data = data.get_train_data()
#test_data = data.get_test_data()

print(type(train_data[0]))
print(len(train_data))
#print(test_data)


#model = FoodSN()

#model.fit(data)

