import sn.py


num_classes_for_Food101: int = 101
data = train_split
model = FoodSN()

model.fit(data)