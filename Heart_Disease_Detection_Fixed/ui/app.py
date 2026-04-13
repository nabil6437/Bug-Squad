import streamlit as st
import pandas as pd
import numpy as np
import plotly .express as px
import plotly .graph_objects as go
from sklearn .metrics import (accuracy_score ,precision_score ,recall_score ,
f1_score ,roc_auc_score ,roc_curve )
import os ,sys ,json

sys .path .insert (0 ,os .path .join (os .path .dirname (__file__ ),'..'))
from rule_based_system .rules import run_expert_system ,KNOWLEDGE_BASE
from ml_model .predict import load_model ,predict

BASE =os .path .join (os .path .dirname (__file__ ),'..')
DATA_PATH =os .path .join (BASE ,'data','raw_data.csv')

st .set_page_config (page_title ="Heart Disease Detection",layout ="wide",
initial_sidebar_state ="expanded")

C ={
"teal":"#0F766E","teal_light":"#14B8A6","teal_bg":"#F0FDFA",
"dark":"#0F172A","slate":"#334155",
"red":"#DC2626","amber":"#D97706","green":"#059669",
"white":"#FFFFFF","gray100":"#F1F5F9","gray200":"#E2E8F0",
"gray400":"#94A3B8","gray600":"#475569",
"blue":"#3B82F6","indigo":"#6366F1",
}

PLOTLY_LAYOUT =dict (
paper_bgcolor ="rgba(0,0,0,0)",plot_bgcolor ="rgba(0,0,0,0)",
font =dict (family ="Inter, Segoe UI, sans-serif",color =C ["slate"],size =12 ),
margin =dict (l =16 ,r =16 ,t =36 ,b =16 ),
xaxis =dict (gridcolor ="rgba(226,232,240,0.4)",zerolinecolor ="rgba(226,232,240,0.5)"),
yaxis =dict (gridcolor ="rgba(226,232,240,0.4)",zerolinecolor ="rgba(226,232,240,0.5)"),
legend =dict (orientation ="h",yanchor ="bottom",y =1.02 ,xanchor ="right",x =1 ),
)

def inject_css ():
    st .markdown (f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .main .block-container {{ max-width: 1140px; padding: 1.5rem 2rem 3rem; }}
    section[data-testid="stSidebar"] {{ width: 240px !important; background: #F8FAFC; border-right: 1px solid {C ['gray200']}; }}
    section[data-testid="stSidebar"] .stRadio > label {{ display: none; }}
    section[data-testid="stSidebar"] .stRadio > div > label {{
        padding: 10px 16px !important; border-radius: 8px; font-weight: 500;
        font-size: 0.88rem; color: {C ['slate']}; transition: all 0.2s; display: block;
    }}
    section[data-testid="stSidebar"] .stRadio > div > label:hover {{ background: {C ['gray100']}; }}
    .card {{ background:{C ['white']}; border:1px solid {C ['gray200']}; border-radius:12px; padding:20px 24px; }}
    .card-shadow {{ box-shadow:0 1px 3px rgba(15,23,42,0.07); }}
    .stat-label {{ font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.7px; color:{C ['gray400']}; margin-bottom:4px; }}
    .stat-value {{ font-size:1.7rem; font-weight:800; color:{C ['dark']}; line-height:1.2; }}
    .stat-sub {{ font-size:0.78rem; color:{C ['gray400']}; margin-top:2px; }}
    .risk-pill {{ display:inline-flex; align-items:center; gap:8px; padding:10px 26px;
        border-radius:40px; font-weight:700; font-size:0.95rem; letter-spacing:0.6px; text-transform:uppercase; }}
    .risk-high {{ background:linear-gradient(135deg,#FEE2E2,#FECACA); color:{C ['red']}; border:1px solid #FECACA; }}
    .risk-medium {{ background:linear-gradient(135deg,#FEF3C7,#FDE68A); color:{C ['amber']}; border:1px solid #FDE68A; }}
    .risk-low {{ background:linear-gradient(135deg,#D1FAE5,#A7F3D0); color:{C ['green']}; border:1px solid #A7F3D0; }}
    .trace-item {{ display:flex; align-items:flex-start; gap:10px; padding:10px 14px;
        background:{C ['gray100']}; border-radius:8px; margin:5px 0; font-size:0.85rem; color:{C ['slate']}; }}
    .trace-dot {{ width:8px; height:8px; min-width:8px; border-radius:50%; margin-top:5px; }}
    .dot-high {{ background:{C ['red']}; }}
    .dot-medium {{ background:{C ['amber']}; }}
    .dot-low {{ background:{C ['green']}; }}
    .rule-row {{ padding:10px 14px; border-bottom:1px solid {C ['gray100']};
        font-size:0.84rem; color:{C ['slate']}; display:flex; gap:12px; align-items:center; }}
    .rule-badge {{ padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:700; letter-spacing:0.5px; text-transform:uppercase; white-space:nowrap; }}
    .badge-high {{ background:#FEE2E2; color:{C ['red']}; }}
    .badge-medium {{ background:#FEF3C7; color:{C ['amber']}; }}
    .badge-low {{ background:#D1FAE5; color:{C ['green']}; }}
    .section-title {{ font-size:1.05rem; font-weight:700; color:{C ['dark']}; margin-bottom:14px; }}
    .page-title {{ font-size:1.6rem; font-weight:800; color:{C ['dark']}; margin-bottom:4px; }}
    .page-sub {{ font-size:0.88rem; color:{C ['gray400']}; margin-bottom:22px; }}
    .divider {{ height:1px; background:{C ['gray200']}; margin:22px 0; }}
    .stButton > button {{ border-radius:8px !important; font-weight:600 !important; transition:all 0.2s !important; }}
    .stButton > button[kind="primary"] {{
        background:linear-gradient(135deg,{C ['teal']},{C ['teal_light']}) !important;
        border:none !important; color:white !important;
        box-shadow:0 2px 8px rgba(15,118,110,0.25) !important;
    }}
    </style>""",unsafe_allow_html =True )

def stat_card (label ,value ,sub ="",delta =""):
    red =C ["red"]
    delta_html =f'<div style="font-size:0.75rem;color:{red };margin-top:2px;">{delta }</div>'if delta else ""
    sub_html =f'<div class="stat-sub">{sub }</div>'if sub else ""
    st .markdown (f"""
    <div class="card card-shadow" style="height:100%">
        <div class="stat-label">{label }</div>
        <div class="stat-value">{value }</div>
        {sub_html }{delta_html }
    </div>""",unsafe_allow_html =True )

def sidebar ():
    with st .sidebar :
        st .markdown (f"""
        <div style="padding:18px 16px 22px;border-bottom:1px solid {C ['gray200']};margin-bottom:12px;">
            <div style="font-size:1.25rem;font-weight:800;color:{C ['dark']};letter-spacing:-0.3px;">Heart Disease</div>
            <div style="font-size:0.7rem;color:{C ['gray400']};text-transform:uppercase;letter-spacing:1px;margin-top:2px;">Detection System</div>
        </div>""",unsafe_allow_html =True )
        page =st .radio ("nav",["Prediction","Expert System","Data Explorer","Model Comparison"],
        label_visibility ="collapsed")
    return page

def build_patient_form_dt ():
    st .markdown ('<div class="page-title">Patient Risk Assessment</div>',unsafe_allow_html =True )
    st .markdown ('<div class="page-sub">Enter patient data for Decision Tree prediction</div>',unsafe_allow_html =True )
    c1 ,c2 ,c3 =st .columns (3 )
    with c1 :
        st .markdown ("**Demographics and Vitals**")
        age =st .number_input ("Age (years)",1 ,120 ,55 ,key ="dt_age")
        sex =st .selectbox ("Sex",[("Female",0 ),("Male",1 )],format_func =lambda x :x [0 ],key ="dt_sex")
        cp =st .selectbox ("Chest Pain",[(0 ,"Typical Angina"),(1 ,"Atypical Angina"),(2 ,"Non-anginal"),(3 ,"Asymptomatic")],
        format_func =lambda x :x [1 ],key ="dt_cp")
        trestbps =st .number_input ("Blood Pressure (mmHg)",80 ,220 ,130 ,key ="dt_bp")
        chol =st .number_input ("Cholesterol (mg/dL)",100 ,600 ,200 ,key ="dt_chol")
    with c2 :
        st .markdown ("**Lab Results**")
        fbs =st .selectbox ("Fasting BS > 120 mg/dL",[("No",0 ),("Yes",1 )],format_func =lambda x :x [0 ],key ="dt_fbs")
        restecg =st .selectbox ("Resting ECG",[(0 ,"Normal"),(1 ,"ST-T Abnormality"),(2 ,"LV Hypertrophy")],
        format_func =lambda x :x [1 ],key ="dt_ecg")
        thalach =st .number_input ("Max Heart Rate (bpm)",60 ,220 ,150 ,key ="dt_hr")
        exang =st .selectbox ("Exercise Angina",[("No",0 ),("Yes",1 )],format_func =lambda x :x [0 ],key ="dt_exang")
        oldpeak =st .number_input ("ST Depression",0.0 ,7.0 ,1.0 ,step =0.1 ,key ="dt_old")
    with c3 :
        st .markdown ("**Cardiac Tests**")
        slope =st .selectbox ("ST Slope",[(0 ,"Downsloping"),(1 ,"Flat"),(2 ,"Upsloping")],
        format_func =lambda x :x [1 ],key ="dt_slope")
        ca =st .number_input ("Major Vessels (0-4)",0 ,4 ,0 ,key ="dt_ca")
        thal =st .selectbox ("Thalassemia",[(0 ,"Unknown"),(1 ,"Fixed Defect"),(2 ,"Normal"),(3 ,"Reversible Defect")],
        format_func =lambda x :x [1 ],key ="dt_thal")
    return {
    "age":age ,"sex":sex [1 ],"cp":cp [0 ],"trestbps":trestbps ,"chol":chol ,
    "fbs":fbs [1 ],"restecg":restecg [0 ],"thalach":thalach ,"exang":exang [1 ],
    "oldpeak":oldpeak ,"slope":slope [0 ],"ca":ca ,"thal":thal [0 ],
    }

def build_patient_form_es ():
    st .markdown ('<div class="page-title">Expert System Diagnosis</div>',unsafe_allow_html =True )
    st .markdown ('<div class="page-sub">Experta forward-chaining inference engine with 20 clinical rules</div>',unsafe_allow_html =True )
    c1 ,c2 =st .columns (2 )
    with c1 :
        st .markdown ("**Demographics and Vitals**")
        age =st .number_input ("Age (years)",1 ,120 ,55 ,key ="es_age")
        sex =st .selectbox ("Sex",[("Female",0 ),("Male",1 )],format_func =lambda x :x [0 ],key ="es_sex")
        trestbps =st .number_input ("Resting Blood Pressure (mmHg)",80 ,220 ,130 ,key ="es_bp")
        chol =st .number_input ("Serum Cholesterol (mg/dL)",100 ,600 ,230 ,key ="es_chol")
        fbs =st .selectbox ("Fasting Blood Sugar > 120 mg/dL",[("No",0 ),("Yes",1 )],format_func =lambda x :x [0 ],key ="es_fbs")
        thalach =st .number_input ("Max Heart Rate Achieved (bpm)",60 ,220 ,150 ,key ="es_hr")
        oldpeak =st .number_input ("ST Depression (oldpeak)",0.0 ,7.0 ,1.0 ,step =0.1 ,key ="es_old")
    with c2 :
        st .markdown ("**Cardiac Tests and ECG**")
        cp =st .selectbox ("Chest Pain Type",[(0 ,"Typical Angina"),(1 ,"Atypical Angina"),(2 ,"Non-anginal"),(3 ,"Asymptomatic")],
        format_func =lambda x :x [1 ],key ="es_cp")
        exang =st .selectbox ("Exercise-Induced Angina",[("No",0 ),("Yes",1 )],format_func =lambda x :x [0 ],key ="es_exang")
        restecg =st .selectbox ("Resting ECG",[(0 ,"Normal"),(1 ,"ST-T Abnormality"),(2 ,"LV Hypertrophy")],
        format_func =lambda x :x [1 ],key ="es_ecg")
        slope =st .selectbox ("ST Segment Slope",[(0 ,"Downsloping"),(1 ,"Flat"),(2 ,"Upsloping")],
        format_func =lambda x :x [1 ],key ="es_slope")
        ca =st .number_input ("Major Vessels (0-4)",0 ,4 ,0 ,key ="es_ca")
        thal =st .selectbox ("Thalassemia Type",[(0 ,"Unknown"),(1 ,"Fixed Defect"),(2 ,"Normal"),(3 ,"Reversible Defect")],
        format_func =lambda x :x [1 ],key ="es_thal")
    return {
    "age":age ,"sex":sex [1 ],"cp":cp [0 ],"trestbps":trestbps ,"chol":chol ,
    "fbs":fbs [1 ],"restecg":restecg [0 ],"thalach":thalach ,"exang":exang [1 ],
    "oldpeak":oldpeak ,"slope":slope [0 ],"ca":ca ,"thal":thal [0 ],
    }

def page_prediction ():
    patient =build_patient_form_dt ()
    st .markdown ("<br>",unsafe_allow_html =True )
    run =st .button ("Run Decision Tree Prediction",type ="primary",use_container_width =True )

    if run :
        st .markdown ('<div class="divider"></div>',unsafe_allow_html =True )
        try :
            model ,scaler ,encoder ,feature_cols =load_model ()
            features =[patient [k ]for k in ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
            pred ,proba =predict (model ,features ,scaler ,encoder ,feature_cols )

            label ="Disease Detected"if pred ==1 else "No Disease"
            css ="risk-high"if pred ==1 else "risk-low"
            icon ="&#x2716;"if pred ==1 else "&#x2714;"
            st .markdown (f'<div class="risk-pill {css }">{icon }&nbsp;{label }</div>',unsafe_allow_html =True )
            st .markdown ("<br>",unsafe_allow_html =True )

            r1 ,r2 =st .columns ([1.2 ,1 ])
            with r1 :
                fig =go .Figure (go .Indicator (
                mode ="gauge+number",
                value =proba [1 ]*100 ,
                number ={"suffix":"%","font":{"size":44 ,"color":C ["dark"],"family":"Inter"}},
                title ={"text":"Disease Probability","font":{"size":13 ,"color":C ["gray400"]}},
                gauge ={
                "axis":{"range":[0 ,100 ],"tickwidth":0 ,"tickcolor":"rgba(0,0,0,0)"},
                "bar":{"color":C ["teal"],"thickness":0.28 },
                "bgcolor":C ["gray100"],"borderwidth":0 ,
                "steps":[
                {"range":[0 ,35 ],"color":"#D1FAE5"},
                {"range":[35 ,65 ],"color":"#FEF3C7"},
                {"range":[65 ,100 ],"color":"#FEE2E2"},
                ],
                },
                ))
                fig .update_layout (height =220 ,margin =dict (l =24 ,r =24 ,t =48 ,b =0 ),paper_bgcolor ="rgba(0,0,0,0)")
                st .plotly_chart (fig ,use_container_width =True )
            with r2 :
                st .markdown ("<br><br>",unsafe_allow_html =True )
                a ,b =st .columns (2 )
                with a :
                    stat_card ("No Disease",f"{proba [0 ]*100 :.1f}%")
                with b :
                    stat_card ("Disease",f"{proba [1 ]*100 :.1f}%")

        except Exception as e :
            st .error (f"Train the model first:\n  python ml_model/train_model.py\n\n{e }")

def page_expert_system ():
    with st .expander ("Knowledge Base — 20 Clinical Rules",expanded =False ):
        level_colors ={"HIGH":"badge-high","MEDIUM":"badge-medium","LOW":"badge-low"}
        for rule in KNOWLEDGE_BASE :
            badge =f'<span class="rule-badge {level_colors [rule ["level"]]}">{rule ["level"]}</span>'
            st .markdown (
            f'<div class="rule-row">'
            f'<b style="color:{C ["dark"]};min-width:60px">Rule {rule ["id"]}</b>'
            f'{badge }'
            f'<span><b>{rule ["name"]}</b> &mdash; <span style="color:{C ["gray400"]}">{rule ["condition"]}</span></span>'
            f'</div>',
            unsafe_allow_html =True
            )
    st .markdown ("<br>",unsafe_allow_html =True )

    patient =build_patient_form_es ()
    st .markdown ("<br>",unsafe_allow_html =True )
    run =st .button ("Run Expert System Diagnosis",type ="primary",use_container_width =True )

    if run :
        st .markdown ('<div class="divider"></div>',unsafe_allow_html =True )
        risk ,reasons ,fired_rules ,score =run_expert_system (patient )

        high_count =sum (1 for r in fired_rules if r ["level"]=="HIGH")
        medium_count =sum (1 for r in fired_rules if r ["level"]=="MEDIUM")
        low_count =sum (1 for r in fired_rules if r ["level"]=="LOW")

        css_map ={"high":"risk-high","medium":"risk-medium","low":"risk-low"}
        icon_map ={"high":"&#x2716;","medium":"&#x26A0;","low":"&#x2714;"}
        label_map ={"high":"HIGH RISK","medium":"MEDIUM RISK","low":"LOW RISK"}
        st .markdown (f'<div class="risk-pill {css_map [risk ]}">{icon_map [risk ]}&nbsp;{label_map [risk ]}</div>',unsafe_allow_html =True )
        st .markdown ("<br>",unsafe_allow_html =True )

        m1 ,m2 ,m3 ,m4 =st .columns (4 )
        with m1 :stat_card ("Risk Level",risk .capitalize ())
        with m2 :stat_card ("Rules Fired",str (len (fired_rules )))
        with m3 :stat_card ("High-Risk Rules",str (high_count ))
        with m4 :stat_card ("Low-Risk Rules",str (low_count ))

        st .markdown ("<br>",unsafe_allow_html =True )
        left ,right =st .columns ([1.1 ,1 ])

        with left :
            st .markdown ('<div class="section-title">Inference Trace — Fired Rules</div>',unsafe_allow_html =True )
            if fired_rules :
                for r in fired_rules :
                    dot_cls ={"HIGH":"dot-high","MEDIUM":"dot-medium","LOW":"dot-low"}.get (r ["level"],"dot-low")
                    st .markdown (
                    f'<div class="trace-item">'
                    f'<div class="trace-dot {dot_cls }"></div>'
                    f'<div><b>Rule {r ["id"]} {r ["level"]}</b><br>'
                    f'<span style="font-size:0.8rem;color:{C ["gray400"]}">{r ["name"]} — {r ["condition"]}</span></div>'
                    f'</div>',
                    unsafe_allow_html =True
                    )
            else :
                st .info ("No rules were triggered.")

        with right :
            st .markdown ('<div class="section-title">Rules Fired by Risk Level</div>',unsafe_allow_html =True )
            bar_data =pd .DataFrame ({
            "Level":["Low Risk","Medium Risk","High Risk"],
            "Count":[low_count ,medium_count ,high_count ],
            "Color":[C ["green"],C ["amber"],C ["red"]],
            })
            fig_bar =go .Figure (go .Bar (
            x =bar_data ["Count"],y =bar_data ["Level"],
            orientation ="h",
            marker_color =bar_data ["Color"].tolist (),
            text =bar_data ["Count"],textposition ="outside",
            ))
            fig_bar .update_layout (**{**PLOTLY_LAYOUT ,"height":180 ,
            "xaxis_title":"Count","yaxis_title":""})
            st .plotly_chart (fig_bar ,use_container_width =True )

            st .markdown ('<div class="section-title">Risk Score</div>',unsafe_allow_html =True )
            fig_g =go .Figure (go .Indicator (
            mode ="gauge+number",
            value =score ,
            number ={"font":{"size":36 ,"color":C ["dark"],"family":"Inter"}},
            gauge ={
            "axis":{"range":[-10 ,20 ]},
            "bar":{"color":C ["teal"],"thickness":0.25 },
            "bgcolor":C ["gray100"],"borderwidth":0 ,
            "steps":[
            {"range":[-10 ,0 ],"color":"#D1FAE5"},
            {"range":[0 ,8 ],"color":"#FEF3C7"},
            {"range":[8 ,20 ],"color":"#FEE2E2"},
            ],
            },
            ))
            fig_g .update_layout (height =200 ,margin =dict (l =24 ,r =24 ,t =24 ,b =0 ),paper_bgcolor ="rgba(0,0,0,0)")
            st .plotly_chart (fig_g ,use_container_width =True )

def page_explorer ():
    st .markdown ('<div class="page-title">Data Explorer</div>',unsafe_allow_html =True )
    st .markdown ('<div class="page-sub">Interactive exploration of the heart disease dataset</div>',unsafe_allow_html =True )
    try :
        df =pd .read_csv (DATA_PATH ).drop_duplicates ().reset_index (drop =True )
        df ["target"]=1 -df ["target"]
    except Exception as e :
        st .error (f"Data not found: {e }");return

    c1 ,c2 ,c3 ,c4 =st .columns (4 )
    with c1 :stat_card ("Patients",f"{len (df ):,}")
    with c2 :stat_card ("Disease Cases",f"{int (df ['target'].sum ()):,}",f"{df ['target'].mean ()*100 :.0f}% of total")
    with c3 :stat_card ("Avg Age",f"{df ['age'].mean ():.0f}","years")
    with c4 :stat_card ("Avg Cholesterol",f"{df ['chol'].mean ():.0f}","mg/dL")

    st .markdown ('<div class="divider"></div>',unsafe_allow_html =True )
    c1 ,c2 =st .columns (2 )
    with c1 :
        st .markdown ('<div class="section-title">Target Distribution</div>',unsafe_allow_html =True )
        counts =df ["target"].value_counts ().reset_index ()
        counts .columns =["target","count"]
        counts ["label"]=counts ["target"].map ({0 :"Healthy",1 :"Disease"})
        fig =go .Figure (go .Pie (
        labels =counts ["label"],values =counts ["count"],hole =0.6 ,
        marker =dict (colors =[C ["teal"],C ["red"]],line =dict (color =C ["white"],width =3 )),
        textinfo ="percent+label",
        ))
        fig .update_layout (**{**PLOTLY_LAYOUT ,"height":300 ,"showlegend":False })
        fig .add_annotation (text =f"<b>{len (df )}</b><br><span style='font-size:11px;color:{C ['gray400']}'>patients</span>",
        x =0.5 ,y =0.5 ,showarrow =False ,font =dict (size =20 ,color =C ["dark"]))
        st .plotly_chart (fig ,use_container_width =True )
    with c2 :
        st .markdown ('<div class="section-title">Age Distribution</div>',unsafe_allow_html =True )
        fig =go .Figure ()
        for val ,name ,color in [(0 ,"Healthy",C ["teal"]),(1 ,"Disease",C ["red"])]:
            s =df [df ["target"]==val ]
            fig .add_trace (go .Histogram (x =s ["age"],name =name ,marker_color =color ,opacity =0.72 ,nbinsx =22 ))
        fig .update_layout (**{**PLOTLY_LAYOUT ,"height":300 ,"barmode":"overlay","xaxis_title":"Age","yaxis_title":"Count"})
        st .plotly_chart (fig ,use_container_width =True )

    st .markdown ('<div class="section-title">Correlation Heatmap</div>',unsafe_allow_html =True )
    corr =df .corr ()
    fig =px .imshow (corr ,text_auto =".2f",aspect ="auto",
    color_continuous_scale =["#0F766E","#99F6E4","#FFFFFF","#FECACA","#DC2626"],zmin =-1 ,zmax =1 )
    fig .update_layout (**{**PLOTLY_LAYOUT ,"height":460 })
    st .plotly_chart (fig ,use_container_width =True )

    c1 ,c2 =st .columns (2 )
    with c1 :
        st .markdown ('<div class="section-title">Feature Importance</div>',unsafe_allow_html =True )
        corr_t =df .corr ()["target"].drop ("target").abs ().sort_values ()
        fig =go .Figure (go .Bar (
        x =corr_t .values ,y =corr_t .index ,orientation ="h",
        marker_color =[C ["teal"]if v >0.15 else C ["gray400"]for v in corr_t .values ],
        text =[f"{v :.3f}"for v in corr_t .values ],textposition ="outside",
        ))
        fig .update_layout (**{**PLOTLY_LAYOUT ,"height":380 ,"xaxis_title":"Absolute Correlation"})
        st .plotly_chart (fig ,use_container_width =True )
    with c2 :
        st .markdown ('<div class="section-title">Feature Explorer</div>',unsafe_allow_html =True )
        feat =st .selectbox ("Select feature",[c for c in df .columns if c !="target"])
        fig =go .Figure ()
        for val ,name ,color in [(0 ,"Healthy",C ["teal"]),(1 ,"Disease",C ["red"])]:
            s =df [df ["target"]==val ]
            fig .add_trace (go .Box (y =s [feat ],name =name ,marker_color =color ,boxmean =True ,line =dict (width =1.5 )))
        fig .update_layout (**{**PLOTLY_LAYOUT ,"height":380 ,"showlegend":False ,"yaxis_title":feat })
        st .plotly_chart (fig ,use_container_width =True )

@st .cache_data
def compute_comparison_metrics ():
    df =pd .read_csv (DATA_PATH ).drop_duplicates ().reset_index (drop =True )
    df ["target"]=1 -df ["target"]

    try :
        model ,scaler ,encoder ,feature_cols =load_model ()
        features_list =["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
        from utils .data_processing import transform_single_patient
        dt_preds ,dt_proba =[],[]
        for _ ,row in df .iterrows ():
            p ={k :row [k ]for k in features_list }
            scaled =transform_single_patient (p ,scaler ,encoder ,feature_cols )
            inp =pd .DataFrame ([scaled ],columns =feature_cols )
            dt_preds .append (int (model .predict (inp )[0 ]))
            dt_proba .append (model .predict_proba (inp )[0 ][1 ])
        dt_acc =accuracy_score (df ["target"],dt_preds )
        dt_prec =precision_score (df ["target"],dt_preds ,zero_division =0 )
        dt_rec =recall_score (df ["target"],dt_preds ,zero_division =0 )
        dt_f1 =f1_score (df ["target"],dt_preds ,zero_division =0 )
        dt_auc =roc_auc_score (df ["target"],dt_proba )
        dt_fpr ,dt_tpr ,_ =roc_curve (df ["target"],dt_proba )
    except Exception :
        dt_acc =dt_prec =dt_rec =dt_f1 =dt_auc =0.0
        dt_fpr =dt_tpr =np .array ([0 ,1 ])

    es_preds ,es_scores =[],[]
    for _ ,row in df .iterrows ():
        patient ={k :row [k ]for k in ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]}
        risk ,_ ,_ ,score =run_expert_system (patient )
        pred =1 if risk =="high"else (1 if risk =="medium"else 0 )
        es_preds .append (pred )
        es_scores .append (score )
    es_acc =accuracy_score (df ["target"],es_preds )
    es_prec =precision_score (df ["target"],es_preds ,zero_division =0 )
    es_rec =recall_score (df ["target"],es_preds ,zero_division =0 )
    es_f1 =f1_score (df ["target"],es_preds ,zero_division =0 )
    es_auc =roc_auc_score (df ["target"],es_scores )if len (set (es_scores ))>1 else 0.5
    es_fpr ,es_tpr ,_ =roc_curve (df ["target"],es_scores )

    return {
    "dt":{"acc":dt_acc ,"prec":dt_prec ,"rec":dt_rec ,"f1":dt_f1 ,"auc":dt_auc ,"fpr":dt_fpr ,"tpr":dt_tpr },
    "es":{"acc":es_acc ,"prec":es_prec ,"rec":es_rec ,"f1":es_f1 ,"auc":es_auc ,"fpr":es_fpr ,"tpr":es_tpr },
    "n_raw":1025 ,"n_clean":len (df ),
    }

def page_comparison ():
    st .markdown ('<div class="page-title">Model Comparison</div>',unsafe_allow_html =True )
    st .markdown ('<div class="page-sub">Decision Tree vs Expert System — performance on 305 patients</div>',unsafe_allow_html =True )

    with st .spinner ("Computing metrics..."):
        m =compute_comparison_metrics ()
    dt ,es =m ["dt"],m ["es"]

    c1 ,c2 ,c3 ,c4 ,c5 =st .columns (5 )
    with c1 :stat_card ("Raw Records",f"{m ['n_raw']:,}")
    with c2 :stat_card ("Duplicates",f"-{m ['n_raw']-m ['n_clean']:,}","","↓ -70%")
    with c3 :stat_card ("Clean Records",f"{m ['n_clean']:,}")
    with c4 :stat_card ("DT Accuracy",f"{dt ['acc']*100 :.1f}%")
    with c5 :stat_card ("AUC-ROC (DT)",f"{dt ['auc']:.3f}")

    st .markdown ('<div class="divider"></div>',unsafe_allow_html =True )

    c1 ,c2 =st .columns (2 )
    with c1 :
        st .markdown ('<div class="section-title">Target Distribution</div>',unsafe_allow_html =True )
        df =pd .read_csv (DATA_PATH ).drop_duplicates ().reset_index (drop =True )
        df ["target"]=1 -df ["target"]
        d_count =int (df ["target"].sum ())
        h_count =len (df )-d_count
        fig =go .Figure (go .Pie (
        labels =["Heart Disease","No Disease"],
        values =[d_count ,h_count ],hole =0.55 ,
        marker =dict (colors =[C ["red"],C ["teal_light"]],line =dict (color =C ["white"],width =3 )),
        textinfo ="percent",textfont =dict (size =13 ),
        ))
        fig .update_layout (**{**PLOTLY_LAYOUT ,"height":280 ,"showlegend":True ,
        "legend":dict (orientation ="v",x =1 ,y =0.5 )})
        fig .add_annotation (text =f"<b>{len (df )}</b><br><span style='font-size:11px'>patients</span>",
        x =0.5 ,y =0.5 ,showarrow =False ,font =dict (size =20 ,color =C ["dark"]))
        st .plotly_chart (fig ,use_container_width =True )

    with c2 :
        st .markdown ('<div class="section-title">Model Accuracy Comparison</div>',unsafe_allow_html =True )
        fig =go .Figure ()
        fig .add_trace (go .Bar (name ="Decision Tree",x =["Decision Tree"],y =[dt ["acc"]*100 ],
        marker_color =C ["indigo"],text =[f"{dt ['acc']*100 :.1f}%"],textposition ="outside",
        textfont =dict (size =14 ,color =C ["dark"])))
        fig .add_trace (go .Bar (name ="Expert System",x =["Expert System"],y =[es ["acc"]*100 ],
        marker_color =C ["red"],opacity =0.8 ,
        text =[f"{es ['acc']*100 :.1f}%"],textposition ="outside",
        textfont =dict (size =14 ,color =C ["dark"])))
        fig .update_layout (**{**PLOTLY_LAYOUT ,"height":280 ,"barmode":"group","showlegend":False ,
        "yaxis":dict (range =[0 ,115 ],title ="Accuracy (%)")})
        st .plotly_chart (fig ,use_container_width =True )

    st .markdown ('<div class="divider"></div>',unsafe_allow_html =True )
    c1 ,c2 =st .columns (2 )

    with c1 :
        st .markdown ('<div class="section-title">Performance Radar</div>',unsafe_allow_html =True )
        cats =["Accuracy","Precision","Recall","F1","AUC"]
        fig =go .Figure ()
        fig .add_trace (go .Scatterpolar (
        r =[dt ["acc"],dt ["prec"],dt ["rec"],dt ["f1"],dt ["auc"]],
        theta =cats ,fill ="toself",
        name ="Decision Tree",line =dict (color =C ["indigo"]),
        fillcolor =f"rgba(99,102,241,0.15)",
        ))
        fig .add_trace (go .Scatterpolar (
        r =[es ["acc"],es ["prec"],es ["rec"],es ["f1"],es ["auc"]],
        theta =cats ,fill ="toself",
        name ="Expert System",line =dict (color =C ["red"]),
        fillcolor =f"rgba(220,38,38,0.15)",
        ))
        fig .update_layout (
        polar =dict (radialaxis =dict (visible =True ,range =[0 ,1 ],tickformat =".0%")),
        paper_bgcolor ="rgba(0,0,0,0)",height =340 ,
        margin =dict (l =40 ,r =40 ,t =40 ,b =40 ),
        legend =dict (orientation ="h",y =-0.1 ,x =0.5 ,xanchor ="center"),
        )
        st .plotly_chart (fig ,use_container_width =True )

    with c2 :
        st .markdown ('<div class="section-title">ROC Curve Comparison</div>',unsafe_allow_html =True )
        fig =go .Figure ()
        fig .add_trace (go .Scatter (x =dt ["fpr"],y =dt ["tpr"],mode ="lines",
        name =f"Decision Tree (AUC={dt ['auc']:.3f})",
        line =dict (color =C ["indigo"],width =2 )))
        fig .add_trace (go .Scatter (x =es ["fpr"],y =es ["tpr"],mode ="lines",
        name =f"Expert System (AUC={es ['auc']:.3f})",
        line =dict (color =C ["red"],width =2 )))
        fig .add_trace (go .Scatter (x =[0 ,1 ],y =[0 ,1 ],mode ="lines",
        name ="Random Classifier",
        line =dict (color =C ["gray400"],width =1 ,dash ="dash")))
        fig .update_layout (**{**PLOTLY_LAYOUT ,"height":340 ,
        "xaxis_title":"FPR","yaxis_title":"TPR",
        "legend":dict (orientation ="v",x =0.55 ,y =0.15 )})
        st .plotly_chart (fig ,use_container_width =True )

    st .markdown ('<div class="divider"></div>',unsafe_allow_html =True )
    st .markdown ('<div class="section-title">Side-by-Side Metrics</div>',unsafe_allow_html =True )
    metrics_names =["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]
    dt_vals =[dt ["acc"],dt ["prec"],dt ["rec"],dt ["f1"],dt ["auc"]]
    es_vals =[es ["acc"],es ["prec"],es ["rec"],es ["f1"],es ["auc"]]
    fig =go .Figure ()
    fig .add_trace (go .Bar (name ="Decision Tree",x =metrics_names ,y =[v *100 for v in dt_vals ],
    marker =dict (color =C ["indigo"]),
    text =[f"{v *100 :.1f}%"for v in dt_vals ],textposition ="outside",
    textfont =dict (size =12 ,color =C ["dark"])))
    fig .add_trace (go .Bar (name ="Expert System",x =metrics_names ,y =[v *100 for v in es_vals ],
    marker =dict (color =C ["red"],opacity =0.8 ),
    text =[f"{v *100 :.1f}%"for v in es_vals ],textposition ="outside",
    textfont =dict (size =12 ,color =C ["dark"])))
    fig .update_layout (**{**PLOTLY_LAYOUT ,"height":360 ,"barmode":"group",
    "yaxis":dict (range =[0 ,115 ],title ="Score (%)")})
    st .plotly_chart (fig ,use_container_width =True )

    st .markdown ('<div class="section-title">Comparison Table</div>',unsafe_allow_html =True )
    table_rows =[
    ("Accuracy",f"{dt ['acc']*100 :.1f}%",f"{es ['acc']*100 :.1f}%"),
    ("AUC-ROC",f"{dt ['auc']:.3f}",f"{es ['auc']:.3f}"),
    ("Training Data","Required (305 rows)","Not required"),
    ("Interpretability","Moderate — feature imp.","Full — rule trace"),
    ("Domain Knowledge","Implicit in data","Built-in (20 rules)"),
    ("Clinical Trust","Moderate","Very High"),
    ("Update Process","Retrain on new data","Manual rule update"),
    ("Speed","Fast","Instant"),
    ]
    header =f"""
    <div style="display:grid;grid-template-columns:1.5fr 1fr 1fr;background:{C ['gray100']};
                border-radius:8px 8px 0 0;padding:10px 16px;font-weight:700;font-size:0.82rem;
                color:{C ['gray400']};text-transform:uppercase;letter-spacing:0.5px;">
        <span>Aspect</span><span>Decision Tree</span><span>Expert System</span>
    </div>"""
    rows_html =""
    for i ,(asp ,dv ,ev )in enumerate (table_rows ):
        bg =C ["white"]if i %2 ==0 else C ["gray100"]
        rows_html +=f"""
        <div style="display:grid;grid-template-columns:1.5fr 1fr 1fr;background:{bg };
                    padding:10px 16px;font-size:0.86rem;color:{C ['slate']};border-bottom:1px solid {C ['gray200']};">
            <span style="font-weight:600;color:{C ['dark']}">{asp }</span>
            <span>{dv }</span><span>{ev }</span>
        </div>"""
    st .markdown (f'<div style="border:1px solid {C ["gray200"]};border-radius:10px;overflow:hidden;">{header }{rows_html }</div>',
    unsafe_allow_html =True )

inject_css ()
page =sidebar ()

if page =="Prediction":
    page_prediction ()
elif page =="Expert System":
    page_expert_system ()
elif page =="Data Explorer":
    page_explorer ()
elif page =="Model Comparison":
    page_comparison ()

st .markdown (f'<div style="text-align:center;padding:32px 0 8px;font-size:0.72rem;color:{C ["gray400"]};">Heart Disease Detection System — Expert Systems Project</div>',
unsafe_allow_html =True )
