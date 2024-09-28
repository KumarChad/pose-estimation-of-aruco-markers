import pickle
with open('dist.pkl', 'rb') as file:
    myobj = pickle.load(file)
    print(myobj)