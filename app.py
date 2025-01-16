from datetime import datetime
import io
from urllib import request
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from flask import jsonify
import pandas as pd
import joblib
import json
from firebase_admin import credentials, db,initialize_app
import firebase_admin
import tensorflow as tf
from PIL import Image
import numpy as np
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# Load environment variables
load_dotenv()

# Fetch variables
# firebase_credentials = os.getenv('FIREBASE_CREDENTIALS')
firebase_db_url = os.getenv('FIREBASE_DB_URL')
groq_key = os.getenv('GROQ_KEY')

firebase_credentials = os.getenv('FIREBASE_CREDENTIALS') 
# Parse the Firebase credentials JSON string
firebase_credentials_dict = json.loads(firebase_credentials)
# Fix the `private_key` by replacing literal '\n' with actual newlines
firebase_credentials_dict['private_key'] = firebase_credentials_dict['private_key'].replace("\\n", "\n")




# cred = credentials.Certificate(firebase_credentials_dict)
# initialize_app(cred, {
#     'databaseURL': firebase_db_url
# })

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_credentials_dict)
    initialize_app(cred, {'databaseURL': firebase_db_url})













# Firebase Admin SDK setup

# diabetesfull = joblib.load('models/diabetesfull.pkl')
# diabeteshalf = joblib.load('models/diabeteshalf.pkl')
# hypertensionfull = joblib.load('models/hypertensionfull.pkl')
# hypertensionhalf = joblib.load('models/hypertensionhalf.pkl')
# hearthalf = joblib.load('models/hearthalf.pkl')
# heartfull = joblib.load('models/heartfull.pkl')
# brainTumor = tf.keras.models.load_model('models/tumor.h5')
# skin = tf.keras.models.load_model('models/skin.h5')
# chestCancer = tf.keras.models.load_model('models/ChestCancer.h5')
# breastCancer = tf.keras.models.load_model('models/BreastCancer.h5')

# Define the base directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for joblib models
diabetesfull_path = os.path.join(BASE_DIR, "models", "diabetesfull.pkl")
diabeteshalf_path = os.path.join(BASE_DIR, "models", "diabeteshalf.pkl")
hypertensionfull_path = os.path.join(BASE_DIR, "models", "hypertensionfull.pkl")
hypertensionhalf_path = os.path.join(BASE_DIR, "models", "hypertensionhalf.pkl")
hearthalf_path = os.path.join(BASE_DIR, "models", "hearthalf.pkl")
heartfull_path = os.path.join(BASE_DIR, "models", "heartfull.pkl")

# Paths for TensorFlow models
brainTumor_path = os.path.join(BASE_DIR, "models", "tumor.h5")
skin_path = os.path.join(BASE_DIR, "models", "skin.h5")
chestCancer_path = os.path.join(BASE_DIR, "models", "ChestCancer.h5")
breastCancer_path = os.path.join(BASE_DIR, "models", "BreastCancer.h5")

# Load joblib models
diabetesfull = joblib.load(diabetesfull_path)
diabeteshalf = joblib.load(diabeteshalf_path)
hypertensionfull = joblib.load(hypertensionfull_path)
hypertensionhalf = joblib.load(hypertensionhalf_path)
hearthalf = joblib.load(hearthalf_path)
heartfull = joblib.load(heartfull_path)

# Load TensorFlow models
brainTumor = tf.keras.models.load_model(brainTumor_path)
skin = tf.keras.models.load_model(skin_path)
chestCancer = tf.keras.models.load_model(chestCancer_path)
breastCancer = tf.keras.models.load_model(breastCancer_path)

# Debugging: Print paths to verify correctness
# print(f"Loaded models from:\n"
#       f" - {diabetesfull_path}\n"
#       f" - {diabeteshalf_path}\n"
#       f" - {hypertensionfull_path}\n"
#       f" - {hypertensionhalf_path}\n"
#       f" - {hearthalf_path}\n"
#       f" - {heartfull_path}\n"
#       f" - {brainTumor_path}\n"
#       f" - {skin_path}\n"
#       f" - {chestCancer_path}\n"
#       f" - {breastCancer_path}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

llm = ChatGroq(
    model = "llama-3.2-90b-vision-preview",
    temperature=0,
    groq_api_key=groq_key
)

def get_llama_response(query,prompt):
    response = llm.invoke([query,prompt])
    return response.content

@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.post('/predict')
def predict():
    return None


@app.post('/storetodb')
def storetodb(data: str = Form(...), userid: str = Form(...)):
    try:
        data_dict = json.loads(data)
        print("Received data:", data_dict)
        
        # Store directly without nested 'data' key
        ref = db.reference(f"data/{userid}")
        ref.set(data_dict)
        
        return JSONResponse(content={"message": "Successful"})
    except Exception as e:
        print("Error in storetodb:", str(e))
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
   
@app.post('/send_message')
def send_message(
    message: str = Form(...),
    userid: str = Form(...),
    community: str = Form(...),
    timestamp: str = Form(...),
    userImageURL: str = Form(...)
):
    ref = db.reference(f"cummunity/{community}")
    message_body = {
        'userid':userid,
        'message':message,
        'timestamp':timestamp,
        'userImageURL':userImageURL
    }
    ref.push(message_body)
    return JSONResponse(content={
        'message':'Successfull'
    })

@app.post('/get_message')
def get_message(
    community: str = Form(...),
):
    ref = db.reference(f"cummunity/{community}")
    data = ref.get()
    print(data)
    return JSONResponse(content={
        'data':data
    })

@app.post('/upload_doc')
async def upload_doc(
    userid: str = Form(...),
    docname: str = Form(...),
    description: str = Form(...),
    link: str = Form(...)
):
    ref = db.reference(f"healerai/docs/{userid}")
    data = {
        'docname':docname,
        'description':description,
        'link':link
    }
    ref.push(data)
    return JSONResponse(content={
        'message':'Successfull'
    })




@app.post('/get_docs')
async def get_docs(
    userid: str = Form(...),
):
    ref = db.reference(f"healerai/docs/{userid}")
    data = ref.get()
    print(data)
    return JSONResponse(content={
        'data':data
    })




@app.post('/hypertension')
async def hypertension(
    gender: str = Form(...),
    age: str = Form(...),
    cigsPerDay: str = Form(...),
    BPMeds: str = Form(...),
    totChol: str = Form(...),
    sysBP: str = Form(...),
    diaBP: str = Form(...),
    weight: str = Form(...),
    height: str = Form(...),
    heartRate: str = Form(...),
    glucose: str = Form(...),
    userid: str = Form(...)
):
    input = {
        'male': int(gender),
        'age': int(age),
        'cigsPerDay': int(cigsPerDay),
        'BPMeds' : int(BPMeds),
        'totChol' : int(totChol),    
        'sysBP' : int(sysBP),
        'diaBP' : int(diaBP),        
        'BMI': int(weight) / ((int(height)/ 100) ** 2),
        'heartRate': int(heartRate),
        'glucose' : int(glucose),
    }

    data = pd.DataFrame([input])

    prediction = hypertensionfull.predict_proba(data)[0]

    ref = db.reference(f"diseaseProbability/{userid}/hypertension")

    data = { 
        'male': int(gender),
        'age': int(age),
        'cigsPerDay': int(cigsPerDay),
        'BPMeds' : int(BPMeds),
        'totChol' : int(totChol),    
        'sysBP' : int(sysBP),
        'diaBP' : int(diaBP),        
        'BMI': int(weight) / ((int(height)/ 100) ** 2),
        'heartRate': int(heartRate),
        'glucose' : int(glucose),
        'probability' : float(prediction[1])}

    ref.set(data)

    return {
        'status': 'success',
    }




@app.post('/diabetes')
async def diabetes(
    gender: str = Form(...),
    age: str = Form(...),
    hyperTension: str = Form(...),
    heartDisease: str = Form(...),
    cigsPerDay: str = Form(...),
    weight: str = Form(...),
    height: str = Form(...),
    hba1c: str = Form(...),
    glucose: str = Form(...),
):
    input = {
        'gender': gender,
        'age': age,
        'hypertension': hyperTension,  # Assuming hyperTension is defined elsewhere
        'heart_disease': heartDisease,
        'smoking_history': 1 if cigsPerDay > 0 else 0,
        'bmi': weight / ((height / 100) ** 2),
        'HbA1c_level' : hba1c,
        'blood_glucose_level' : glucose
    }

    data = pd.DataFrame([input])

    prediction = diabetesfull.predict_proba(data)[0]

    ref = db.reference(f"diseaseProbability/")

    data = { 'heartdisease' : float(prediction[1])}

    ref.push(data)

    return {
        'status': 'success',
    }





@app.post('/heartdisease')
async def heartdisease(
    gender: str = Form(...),
    age: str = Form(...),
    cigsPerDay: str = Form(...),
    cholesterol: str = Form(...),
    weight: str = Form(...),
    height: str = Form(...),
    glucose: str = Form(...),
    ap_lo: str = Form(...),
    ap_hi: str = Form(...),
    alco: str = Form(...),
    active: str = Form(...),
    userid: str = Form(...)
):
    input = {
        'age': int(age)*365, 
        'gender': int(gender),
        'height': int(height),
        'weight': int(weight),
        'ap_hi' : int(ap_hi),
        'ap_lo' : int(ap_lo),
        'cholesterol' : int(cholesterol),  
        'gluc' : int(glucose), 
        'smoke': 1 if int(cigsPerDay) > 0 else 0,
        'alco': int(alco),
        'active': int(active),
        'bmi': int(weight) / ((int(height) / 100) ** 2)
    }

    data = pd.DataFrame([input])

    prediction = heartfull.predict_proba(data)[0]

    ref = db.reference(f"diseaseProbability/{userid}/heartDisease")

    data = {
        'age': int(age)*365, 
        'gender': int(gender),
        'height': int(height),
        'weight': int(weight),
        'ap_hi' : int(ap_hi),
        'ap_lo' : int(ap_lo),
        'cholesterol' : int(cholesterol),  
        'gluc' : int(glucose), 
        'smoke': 1 if int(cigsPerDay) > 0 else 0,
        'alco': int(alco),
        'active': int(active),
        'bmi': int(weight) / ((int(height) / 100) ** 2),
        'probability' : float(prediction[1])
    }

    ref.set(data)

    return {
        'status': 'success',
    }




@app.post('/braintumor')
async def braintumor(
    image : UploadFile = File(...),
    userid : str = Form(...),
    image_url : str = Form(...)
):
    names = {0:'Glioma', 1:'Healthy', 2:'Meningioma', 3:'Pituitary'}

    image = await image.read()

    image = Image.open(io.BytesIO(image))

    resized_image = image.resize((256,256))

    img_array = np.array(resized_image)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = brainTumor.predict(img_array)

    top_class_index = np.argmax(prediction)

    predicted_class = names[top_class_index]

    probability = prediction[0, top_class_index] 

    print('Probability:', probability)

    print(predicted_class)

    ref = db.reference(f"diseaseProbability/{userid}/brainTumor")

    data = {
        'probability' : float(probability),
        'predicted_class' : predicted_class,
        'image_url' : image_url
    }

    ref.set(data)

    return {
        'status': 'success',
    }





@app.post('/chestcancer')
async def chestcancer(
    image : UploadFile = File(...),
    userid : str = Form(...),
    image_url : str = Form(...)
):
    names = {0:'Adenocarcinoma', 1:'Lrge cell Carcinoma', 2:'Normal', 3:'Squamous Cell Carcinoma'}

    image = await image.read()

    image = Image.open(io.BytesIO(image))

    resized_image = image.resize((256,256))

    img_array = np.array(resized_image)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = chestCancer.predict(img_array)

    top_class_index = np.argmax(prediction)

    predicted_class = names[top_class_index]

    probability = prediction[0, top_class_index] 

    print('Probability:', probability)

    print(predicted_class)

    ref = db.reference(f"diseaseProbability/{userid}/chestcancer")

    data = {
        'probability' : float(probability),
        'predicted_class' : predicted_class,
        'image_url' : image_url
    }

    ref.set(data)

    return {
        'status': 'success',
    }








@app.post('/breastcancer')
async def breastcancer(
    image : UploadFile = File(...),
    userid : str = Form(...),
    image_url : str = Form(...)
):
    names = {0:'Normal', 1:'Breast Cancer'}

    image = await image.read()

    image = Image.open(io.BytesIO(image))

    resized_image = image.resize((256,256))

    img_array = np.array(resized_image)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = breastCancer.predict(img_array)

    top_class_index = np.argmax(prediction)

    predicted_class = names[top_class_index]

    probability = prediction[0, top_class_index] 

    print('Probability:', probability)

    print(predicted_class)

    ref = db.reference(f"diseaseProbability/{userid}/breastcancer")

    data = {
        'probability' : float(probability),
        'predicted_class' : predicted_class,
        'image_url' : image_url
    }

    ref.set(data)

    return {
        'status': 'success',
    }






@app.post('/skindisease')
async def skindisease(
    image : UploadFile = File(...),
    userid : str = Form(...),
    image_url : str = Form(...)
):
    names = {0: 'Cellulitis', 1: 'Impetigo', 2: 'Athletes Foot', 3: 'Nail Fungus', 4: 'Ringworm', 5: 'Cutaneous Larva Migrans', 6: 'Chickenpox',7:'Measles',8:'Monkeypox' ,9:'Shingles'}

    image = await image.read()

    image = Image.open(io.BytesIO(image))

    resized_image = image.resize((256,256))

    img_array = np.array(resized_image)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = skin.predict(img_array)

    top_class_index = np.argmax(prediction)

    predicted_class = names[top_class_index]

    probability = prediction[0, top_class_index] 

    print('Probability:', probability)

    print(predicted_class)

    ref = db.reference(f"diseaseProbability/{userid}/skindisease")

    data = {
        'probability' : float(probability),
        'predicted_class' : predicted_class,
        'image_url' : image_url
    }

    ref.set(data)

    return {
        'status': 'success',
    }







@app.post('/get_disease_probability')
async def get_disease_probability(
    userid: str = Form(...),
):
    ref = db.reference(f"diseaseProbability/{userid}")
    data = ref.get()
    print(data)
    return JSONResponse(content={
        'data':data
    })






@app.post('/manage_prescription')
async def manage_prescription(
    userid: str = Form(...),
    doctor_id: str = Form(...),
    diagnosis_result: str = Form(...),
    medicines: str = Form(...),
    description: str = Form(...),
):
    ref = db.reference(f"prescriptions/{userid}")
    data = {
        'userid': userid,
        'doctor_id': doctor_id,
        'diagnosis_result': diagnosis_result,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'prescriptionid': f"RX{int(datetime.now().timestamp())}{userid}",
        'medicines': medicines,
        'description': description
    }   

    ref.push(data)
    return JSONResponse(content={
        'message':'Successfull'
    })




@app.post('/get_prescriptions')
async def get_prescriptions(
    userid: str = Form(...),
    doctor_id: str = Form(...)
):
    ref = db.reference(f"prescriptions/{userid}")

    data = ref.get()

    if data:
        filtered_data = {key: value for key, value in data.items() if value.get('doctor_id') == doctor_id}
    else:
        filtered_data = {}

    return JSONResponse(content={
        'data': filtered_data
    })

# @app.post("/ai_agent")
# async def ai_agent(query: str = Form(...), prompt: str = Form(...)):
#     # Use your ChatGroq LLM to get a response
#     answer = get_llama_response(query, prompt)
#     return JSONResponse(content={"answer": answer})





# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




