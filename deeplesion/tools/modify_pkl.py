import pickle

f=open('train_ann.pkl', 'rb')
data= pickle.load(f)
debug_data=data[:100]
f2=open('train_ann.pkl','wb')
pickle.dump(debug_data,f2)
a=1

