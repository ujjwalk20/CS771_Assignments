import pickle as pkl
from sklearn.preprocessing import StandardScaler

# Define your prediction method here
# df is a dataframe containing timestamps, weather data and potentials
def my_predict( df ):
	
	for i in range(df.shape[0]):
  		st = df.iloc[i,0]
  		df.iloc[i,0]=abs(int(st[-8:-6])*100+int(st[-5:-3])-1300)
	
	scaler = StandardScaler()
	scaler.mean_ = [614.5676, 31.7178775, 85.56402, 189.52175, 193.76275, 199.3749, 190.47395]
	scaler.var_ = [1.24214948e+05, 1.17804978e+01, 5.25595989e+02, 4.26670927e+02, 3.40899062e+02, 5.75979550e+02, 3.73898521e+02]
	scaler.scale_ = [352.44141092,   3.43227297,  22.92588034,  20.6560143 ,18.46345207,  23.99957395,  19.33645576]
	df = scaler.transform(df)
	
	with open( "model1.pkl", "rb" ) as file:
		model1 = pkl.load( file )
	with open( "model2.pkl", "rb" ) as file:
		model2 = pkl.load( file )
	
	pred1 = model1.predict(df)
	pred2 = model2.predict(df)
	
	return ( pred1, pred2 )