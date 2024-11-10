from flask import Flask,render_template,url_for,request,jsonify
import joblib
scaler=joblib.load('standardscaler.pkl','rb')
kmeans=joblib.load('model.pkl','rb')
import pickle 
dt=pickle.load(open('labeled_output.pkl','rb'))
# print(df.shape)
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        n=int(request.form['nitrogen'])
        p=int(request.form['phosphorus'])
        k=int(request.form['potassium'])
        t=float(request.form['temperature'])
        h=float(request.form['humidity'])
        ph=float(request.form['ph'])
        r=float(request.form['rainfall'])
        
        user_data=[[n,p,k,t,h,ph,r]]
        # trans_data=scaler.transform(user_data)
        user_data=scaler.transform(user_data)
        prediction=kmeans.predict(user_data)

        
        # print(prediction)
        # dt=dict(df[df['cluster_12']==prediction[0]]['label'].value_counts())
        for key,val in dt.items():
            if val==prediction :
                ls=key
        ls=ls.capitalize()
        # return f'predicted crop according to the given condition is {ls}'
         

        # ls=list(dt.keys())
        # return f'predicted crop is {ls}'
        return render_template('result.html',prediction=ls)
    
if __name__ == '__main__':
    app.run(debug=True)
