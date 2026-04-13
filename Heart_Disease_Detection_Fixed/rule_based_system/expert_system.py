import sys ,os
sys .path .insert (0 ,os .path .join (os .path .dirname (__file__ ),'..'))
from rule_based_system .rules import run_expert_system

def get_patient_input ():
    print ("=== Heart Disease Risk Assessment (Expert System) ===")
    data ={}
    data ['age']=int (input ("Age: "))
    data ['sex']=int (input ("Sex (1=male, 0=female): "))
    data ['cp']=int (input ("Chest pain type (0-3): "))
    data ['trestbps']=int (input ("Resting blood pressure: "))
    data ['chol']=int (input ("Cholesterol: "))
    data ['fbs']=int (input ("Fasting blood sugar > 120 (1=yes, 0=no): "))
    data ['restecg']=int (input ("Resting ECG (0-2): "))
    data ['thalach']=int (input ("Max heart rate achieved: "))
    data ['exang']=int (input ("Exercise-induced angina (1=yes, 0=no): "))
    data ['oldpeak']=float (input ("ST depression (oldpeak): "))
    data ['slope']=int (input ("Slope of ST segment (0-2): "))
    data ['ca']=int (input ("Number of major vessels (0-4): "))
    data ['thal']=int (input ("Thalassemia (0-3): "))
    return data

if __name__ =="__main__":
    patient =get_patient_input ()

    risk ,reasons ,fired_rules ,score =run_expert_system (patient )
    print (f"\nRisk Level : {risk .upper ()}")
    print (f"Risk Score : {score }")
    print ("Reasons:")
    for r in reasons :
        print (f"  - {r }")
