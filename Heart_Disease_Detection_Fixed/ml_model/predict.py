import joblib ,pandas as pd ,os ,sys ,json

BASE =os .path .join (os .path .dirname (__file__ ),'..')
sys .path .insert (0 ,BASE )
from utils .data_processing import FEATURE_COLS ,transform_single_patient

MODEL_PATH =os .path .join (os .path .dirname (__file__ ),'decision_tree_model.pkl')
SCALER_PATH =os .path .join (os .path .dirname (__file__ ),'scaler.pkl')
ENCODER_PATH =os .path .join (os .path .dirname (__file__ ),'ohe_encoder.pkl')
META_PATH =os .path .join (os .path .dirname (__file__ ),'model_meta.json')

def load_model ():

    if not os .path .exists (MODEL_PATH ):
        raise FileNotFoundError ("Model not found. Run: python ml_model/train_model.py")

    model =joblib .load (MODEL_PATH )
    scaler =joblib .load (SCALER_PATH )if os .path .exists (SCALER_PATH )else None
    encoder =joblib .load (ENCODER_PATH )if os .path .exists (ENCODER_PATH )else None

    with open (META_PATH )as f :
        meta =json .load (f )
    feature_cols =meta ['feature_cols']

    return model ,scaler ,encoder ,feature_cols

def predict (model ,features ,scaler =None ,encoder =None ,feature_cols =None ):

    if len (features )!=len (FEATURE_COLS ):
        raise ValueError (f"Expected {len (FEATURE_COLS )} features, got {len (features )}")

    patient_dict =dict (zip (FEATURE_COLS ,features ))

    if scaler is not None and encoder is not None and feature_cols is not None :
        scaled_features =transform_single_patient (patient_dict ,scaler ,encoder ,feature_cols )
    else :
        scaled_features =features
        feature_cols =feature_cols or FEATURE_COLS

    input_df =pd .DataFrame ([scaled_features ],columns =feature_cols )
    prediction =int (model .predict (input_df )[0 ])
    proba =model .predict_proba (input_df )[0 ]
    return prediction ,proba

if __name__ =="__main__":
    model ,scaler ,encoder ,feature_cols =load_model ()
    print ("Heart Disease Prediction  (Decision Tree)")
    print ("="*44 )
    print ("أدخل 13 قيمة بالترتيب:")
    for i ,col in enumerate (FEATURE_COLS ,1 ):
        print (f"  {i :2d}. {col }")
    raw =input ("\nالقيم (مفصولة بفاصلة): ")
    values =[float (v .strip ())for v in raw .split (',')]
    pred ,proba =predict (model ,values ,scaler ,encoder ,feature_cols )
    label ="⚠  Heart Disease DETECTED"if pred ==1 else "✓  No Heart Disease"
    print (f"\nالنتيجة    : {label }")
    print (f"P(مرض)     : {proba [1 ]*100 :.1f}%")
    print (f"P(سليم)    : {proba [0 ]*100 :.1f}%")
