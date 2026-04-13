from experta import *

KNOWLEDGE_BASE =[
{"id":"01","name":"Exercise-Induced Angina","level":"HIGH","condition":"exang = 1"},
{"id":"02","name":"Multiple Blocked Vessels","level":"HIGH","condition":"ca >= 2"},
{"id":"03","name":"Abnormal Thalassemia Pattern","level":"HIGH","condition":"thal = 3 and age > 50"},
{"id":"04","name":"Severe ST Depression","level":"HIGH","condition":"oldpeak > 2.5"},
{"id":"05","name":"High Cholesterol + Elderly Male","level":"HIGH","condition":"chol > 240 and age > 50 and sex = 1"},
{"id":"06","name":"Hypertension + Male","level":"HIGH","condition":"trestbps > 140 and sex = 1"},
{"id":"07","name":"Angina + Low Max HR","level":"HIGH","condition":"exang = 1 and thalach < 120"},
{"id":"08","name":"Flat Slope + ST Depression","level":"HIGH","condition":"slope = 1 and oldpeak > 1.5"},
{"id":"09","name":"High Blood Sugar + Cholesterol","level":"HIGH","condition":"fbs = 1 and chol > 200"},
{"id":"10","name":"Downsloping ST + Angina","level":"HIGH","condition":"slope = 0 and exang = 1"},
{"id":"11","name":"Abnormal ECG + Hypertension","level":"MEDIUM","condition":"restecg >= 1 and trestbps > 130"},
{"id":"12","name":"Thalassemia + Middle Age","level":"MEDIUM","condition":"thal >= 2 and age > 45"},
{"id":"13","name":"Elderly + Low Max HR","level":"MEDIUM","condition":"age > 60 and thalach < 140"},
{"id":"14","name":"Moderate ST + Vessel Blockage","level":"MEDIUM","condition":"oldpeak > 1.0 and ca >= 1"},
{"id":"15","name":"Asymptomatic Chest Pain + Age","level":"MEDIUM","condition":"cp = 0 and age > 55"},
{"id":"16","name":"Young + Normal Vitals","level":"LOW","condition":"age < 40 and chol < 200 and trestbps < 120"},
{"id":"17","name":"High HR + No Angina + Low ST","level":"LOW","condition":"thalach > 160 and exang = 0 and oldpeak < 1.0"},
{"id":"18","name":"Normal Vessels + Normal Thal","level":"LOW","condition":"ca = 0 and thal = 2"},
{"id":"19","name":"Female + Low BP + High HR","level":"LOW","condition":"sex = 0 and trestbps < 120 and thalach > 150"},
{"id":"20","name":"No Risk Factors Detected","level":"LOW","condition":"chol < 180 and trestbps < 120 and exang = 0"},
]

class HeartDiseaseRisk (Fact ):
    pass

class HeartDiseaseExpert (KnowledgeEngine ):
    def __init__ (self ):
        super ().__init__ ()
        self .risk_score =0
        self .fired_rules =[]

    def _fire (self ,rule_id ,level ):
        info =next ((r for r in KNOWLEDGE_BASE if r ["id"]==rule_id ),None )
        if info :
            self .fired_rules .append ({"id":rule_id ,"name":info ["name"],"level":level ,"condition":info ["condition"]})
        delta ={"HIGH":3 ,"MEDIUM":1 ,"LOW":-2 }
        self .risk_score +=delta .get (level ,0 )

    @Rule (HeartDiseaseRisk (exang =L (1 )))
    def rule_01 (self ):
        self ._fire ("01","HIGH")

    @Rule (HeartDiseaseRisk (ca =P (lambda x :x >=2 )))
    def rule_02 (self ):
        self ._fire ("02","HIGH")

    @Rule (HeartDiseaseRisk (thal =L (3 ),age =P (lambda x :x >50 )))
    def rule_03 (self ):
        self ._fire ("03","HIGH")

    @Rule (HeartDiseaseRisk (oldpeak =P (lambda x :x >2.5 )))
    def rule_04 (self ):
        self ._fire ("04","HIGH")

    @Rule (HeartDiseaseRisk (chol =P (lambda x :x >240 ),age =P (lambda x :x >50 ),sex =L (1 )))
    def rule_05 (self ):
        self ._fire ("05","HIGH")

    @Rule (HeartDiseaseRisk (trestbps =P (lambda x :x >140 ),sex =L (1 )))
    def rule_06 (self ):
        self ._fire ("06","HIGH")

    @Rule (HeartDiseaseRisk (exang =L (1 ),thalach =P (lambda x :x <120 )))
    def rule_07 (self ):
        self ._fire ("07","HIGH")

    @Rule (HeartDiseaseRisk (slope =L (1 ),oldpeak =P (lambda x :x >1.5 )))
    def rule_08 (self ):
        self ._fire ("08","HIGH")

    @Rule (HeartDiseaseRisk (fbs =L (1 ),chol =P (lambda x :x >200 )))
    def rule_09 (self ):
        self ._fire ("09","HIGH")

    @Rule (HeartDiseaseRisk (slope =L (0 ),exang =L (1 )))
    def rule_10 (self ):
        self ._fire ("10","HIGH")

    @Rule (HeartDiseaseRisk (restecg =P (lambda x :x >=1 ),trestbps =P (lambda x :x >130 )))
    def rule_11 (self ):
        self ._fire ("11","MEDIUM")

    @Rule (HeartDiseaseRisk (thal =P (lambda x :x >=2 ),age =P (lambda x :x >45 )))
    def rule_12 (self ):
        self ._fire ("12","MEDIUM")

    @Rule (HeartDiseaseRisk (age =P (lambda x :x >60 ),thalach =P (lambda x :x <140 )))
    def rule_13 (self ):
        self ._fire ("13","MEDIUM")

    @Rule (HeartDiseaseRisk (oldpeak =P (lambda x :x >1.0 ),ca =P (lambda x :x >=1 )))
    def rule_14 (self ):
        self ._fire ("14","MEDIUM")

    @Rule (HeartDiseaseRisk (cp =L (0 ),age =P (lambda x :x >55 )))
    def rule_15 (self ):
        self ._fire ("15","MEDIUM")

    @Rule (HeartDiseaseRisk (age =P (lambda x :x <40 ),chol =P (lambda x :x <200 ),trestbps =P (lambda x :x <120 )))
    def rule_16 (self ):
        self ._fire ("16","LOW")

    @Rule (HeartDiseaseRisk (thalach =P (lambda x :x >160 ),exang =L (0 ),oldpeak =P (lambda x :x <1.0 )))
    def rule_17 (self ):
        self ._fire ("17","LOW")

    @Rule (HeartDiseaseRisk (ca =L (0 ),thal =L (2 )))
    def rule_18 (self ):
        self ._fire ("18","LOW")

    @Rule (HeartDiseaseRisk (sex =L (0 ),trestbps =P (lambda x :x <120 ),thalach =P (lambda x :x >150 )))
    def rule_19 (self ):
        self ._fire ("19","LOW")

    @Rule (HeartDiseaseRisk (chol =P (lambda x :x <180 ),trestbps =P (lambda x :x <120 ),exang =L (0 )))
    def rule_20 (self ):
        self ._fire ("20","LOW")

def run_expert_system (patient_data ):
    engine =HeartDiseaseExpert ()
    engine .reset ()
    engine .declare (HeartDiseaseRisk (**patient_data ))
    engine .run ()

    score =engine .risk_score
    fired =engine .fired_rules

    high_fired =any (r ["level"]=="HIGH"for r in fired )
    medium_fired =any (r ["level"]=="MEDIUM"for r in fired )

    if high_fired :
        level ="high"
    elif medium_fired :
        level ="medium"
    else :
        level ="low"

    if not fired :
        fired .append ({"id":"00","name":"No specific rule triggered","level":"LOW","condition":"default fallback"})

    reasons =[r ["name"]+" — "+r ["condition"]for r in fired ]
    return level ,reasons ,fired ,score
