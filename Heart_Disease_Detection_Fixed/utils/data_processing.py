import pandas as pd
import numpy as np
from sklearn .preprocessing import MinMaxScaler ,OneHotEncoder
import os ,json

FEATURE_COLS =['age','sex','cp','trestbps','chol','fbs',
'restecg','thalach','exang','oldpeak','slope','ca','thal']

CONTINUOUS_COLS =['age','trestbps','chol','thalach','oldpeak']

BINARY_COLS =['sex','fbs','exang']

OHE_COLS =['cp','restecg','slope','ca','thal']

CAT_VALID ={
'sex':{0 ,1 },
'cp':{0 ,1 ,2 ,3 },
'fbs':{0 ,1 },
'restecg':{0 ,1 ,2 },
'exang':{0 ,1 },
'slope':{0 ,1 ,2 },
'ca':{0 ,1 ,2 ,3 ,4 },
'thal':{0 ,1 ,2 ,3 },
}

def step1_load (path :str )->pd .DataFrame :

    df =pd .read_csv (path )
    print (f"[1] تحميل: {len (df )} صف، {len (df .columns )} عمود")
    return df

def step2_remove_duplicates (df :pd .DataFrame )->pd .DataFrame :

    before =len (df )
    df =df .drop_duplicates ().reset_index (drop =True )
    removed =before -len (df )
    print (f"[2] إزالة التكرار: حُذف {removed } صف ← تبقى {len (df )} صف")
    return df

def step3_fix_target (df :pd .DataFrame )->pd .DataFrame :

    df =df .copy ()
    df ['target']=1 -df ['target']
    disease =int (df ['target'].sum ())
    healthy =int ((df ['target']==0 ).sum ())
    print (f"[3] تصحيح target: مرضى={disease } | أصحاء={healthy } | نسبة المرض={disease /len (df )*100 :.1f}%")
    return df

def step4_handle_missing (df :pd .DataFrame )->pd .DataFrame :

    df =df .copy ()
    filled =[]
    for col in df .columns :
        n_miss =df [col ].isnull ().sum ()
        if n_miss ==0 :
            continue
        if col in CONTINUOUS_COLS :
            val =df [col ].median ()
            df [col ]=df [col ].fillna (val )
            filled .append (f"{col }({n_miss } → median={val :.2f})")
        else :
            val =df [col ].mode ()[0 ]
            df [col ]=df [col ].fillna (val )
            filled .append (f"{col }({n_miss } → mode={val })")
    if filled :
        print (f"[4] ملء Missing: {', '.join (filled )}")
    else :
        print ("[4] Missing Values: لا يوجد ✓")
    return df

def step5_validate_categoricals (df :pd .DataFrame )->pd .DataFrame :

    df =df .copy ()
    removed_total =0
    for col ,valid_set in CAT_VALID .items ():
        if col not in df .columns :
            continue
        mask_invalid =~df [col ].isin (valid_set )
        n_invalid =mask_invalid .sum ()
        if n_invalid >0 :
            df =df [~mask_invalid ].reset_index (drop =True )
            removed_total +=n_invalid
            print (f"[5] {col }: حُذف {n_invalid } صف بقيم غير متوقعة")
    if removed_total ==0 :
        print ("[5] التحقق من الفئات: كل القيم صحيحة ✓")
    return df

def step6_cap_outliers (df :pd .DataFrame )->pd .DataFrame :

    df =df .copy ()
    report =[]
    for col in CONTINUOUS_COLS :
        Q1 =df [col ].quantile (0.25 )
        Q3 =df [col ].quantile (0.75 )
        IQR =Q3 -Q1
        lo =Q1 -1.5 *IQR
        hi =Q3 +1.5 *IQR
        n_out =int (((df [col ]<lo )|(df [col ]>hi )).sum ())
        if n_out >0 :
            df [col ]=df [col ].clip (lower =lo ,upper =hi )
            report .append (f"{col }({n_out } قيمة → [{lo :.1f},{hi :.1f}])")
    if report :
        print (f"[6] Outlier Capping: {' | '.join (report )}")
    else :
        print ("[6] Outliers: لا يوجد ✓")
    return df

def step7_ohe (df :pd .DataFrame ,encoder =None ,fit :bool =True ):

    df =df .copy ()
    if fit :
        encoder =OneHotEncoder (sparse_output =False ,handle_unknown ='ignore',dtype =int )
        ohe_arr =encoder .fit_transform (df [OHE_COLS ])
    else :
        if encoder is None :
            raise ValueError ("يجب تمرير encoder مدرَّب عند fit=False")
        ohe_arr =encoder .transform (df [OHE_COLS ])

    ohe_names =encoder .get_feature_names_out (OHE_COLS ).tolist ()
    df_ohe =pd .DataFrame (ohe_arr ,columns =ohe_names ,index =df .index )
    df =pd .concat ([df .drop (columns =OHE_COLS ),df_ohe ],axis =1 )

    print (f"[7] One-Hot Encoding: {OHE_COLS } → {len (ohe_names )} عمود جديد "
    f"(إجمالي features: {len (df .columns )-1 })")
    return df ,encoder

def step8_scale_features (df :pd .DataFrame ,scaler =None ,fit :bool =True ):

    df =df .copy ()
    if fit :
        scaler =MinMaxScaler ()
        df [CONTINUOUS_COLS ]=scaler .fit_transform (df [CONTINUOUS_COLS ])
        print (f"[8] Scaling (fit+transform) على: {CONTINUOUS_COLS }")
    else :
        if scaler is None :
            raise ValueError ("يجب تمرير scaler مدرَّب عند fit=False")
        df [CONTINUOUS_COLS ]=scaler .transform (df [CONTINUOUS_COLS ])
        print (f"[8] Scaling (transform only) على: {CONTINUOUS_COLS }")
    return df ,scaler

def step9_feature_correlation (df :pd .DataFrame )->pd .Series :

    corr =df .corr ()['target'].drop ('target').abs ().sort_values (ascending =False )
    print ("[9] أهم الـ Features (top 10 بعد OHE):")
    for feat ,val in corr .head (10 ).items ():
        bar ="█"*int (val *20 )
        print (f"     {feat :20s}: {val :.3f}  {bar }")
    return corr

def feature_selection (df :pd .DataFrame ,target_col :str ='target',
threshold :float =0.05 ):

    corr =df .corr ()[target_col ].drop (target_col ).abs ().sort_values (ascending =False )
    selected =corr [corr >=threshold ].index .tolist ()
    return selected ,corr

def run_pipeline (raw_path :str ,cleaned_path :str =None ,scale :bool =True ):

    print ("="*52 )
    print ("  Heart Disease — Preprocessing Pipeline")
    print ("="*52 )

    df =step1_load (raw_path )
    df =step2_remove_duplicates (df )
    df =step3_fix_target (df )
    df =step4_handle_missing (df )
    df =step5_validate_categoricals (df )
    df =step6_cap_outliers (df )

    df_clean =df .copy ()

    if scale :
        df_ohe ,encoder =step7_ohe (df_clean ,fit =True )
        df_scaled ,scaler =step8_scale_features (df_ohe ,fit =True )
    else :
        df_scaled ,scaler ,encoder =df_clean .copy (),None ,None

    step9_feature_correlation (df_scaled )

    if cleaned_path :
        os .makedirs (os .path .dirname (cleaned_path ),exist_ok =True )
        df_scaled .to_csv (cleaned_path ,index =False )
        print (f"\n✓ Cleaned data saved → {cleaned_path }")
        print (f"  Shape: {df_scaled .shape }")

    print ("="*52 )
    return df_clean ,df_scaled ,scaler ,encoder

def transform_single_patient (patient_dict :dict ,scaler :MinMaxScaler ,
encoder :OneHotEncoder ,feature_cols :list )->list :

    row =pd .DataFrame ([patient_dict ],columns =FEATURE_COLS )

    ohe_arr =encoder .transform (row [OHE_COLS ])
    ohe_names =encoder .get_feature_names_out (OHE_COLS ).tolist ()
    df_ohe =pd .DataFrame (ohe_arr ,columns =ohe_names )
    row =pd .concat ([row .drop (columns =OHE_COLS ).reset_index (drop =True ),df_ohe ],axis =1 )

    row [CONTINUOUS_COLS ]=scaler .transform (row [CONTINUOUS_COLS ])

    return row [feature_cols ].values [0 ].tolist ()

if __name__ =="__main__":
    BASE =os .path .join (os .path .dirname (__file__ ),'..')
    raw =os .path .join (BASE ,'data','raw_data.csv')
    cleaned =os .path .join (BASE ,'data','cleaned_data.csv')
    run_pipeline (raw ,cleaned ,scale =True )
