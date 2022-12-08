import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

model_xg = data["model"]
le_MDVP_Fo = data["MDVP:Fo(Hz)"]
le_MDVP_Fhi = data["MDVP:Fhi(Hz)"]
le_MDVP_Flo = data["MDVP:Flo(Hz)"]
le_MDVP_JitterPer = data["MDVP:Jitter(%)"]
le_MDVP_JitterAbs = data["MDVP:Jitter(Abs)"]
le_MDVP_RAP = data["MDVP:RAP"]
le_MDVP_PPQ = data["MDVP:PPQ"]
le_Jitter_DDP = data["Jitter:DDP"]
le_MDVP_Shimmer = data["MDVP:Shimmer"]
le_MDVP_Shimmer_dB = data["MDVP:Shimmer(dB)"]
le_Shimmer_APQ3 = data["Shimmer:APQ3"]
le_Shimmer_APQ5 = data["Shimmer:APQ5"]
le_MDVP_APQ = data["MDVP:APQ"]
le_Shimmer_DDA = data["Shimmer:DDA"]
le_NHR = data["NHR"]
le_HNR = data["HNR"]
le_RPDE = data["RPDE"]
le_DFA = data["DFA"]
le_spread1 = data["spread1"]
le_spread2 = data["spread2"]
le_D2 = data["D2"]
le_PPE = data["PPE"]


def show_predict_page():
    st.title("Parkinson Disease Prediction")
    st.write("""### We need some information to predict the parkinson disease""")
    MDVP_Fo = st.text_input("MDVP:Fo(Hz)")
    MDVP_Fhi = st.text_input("MDVP:Fhi(Hz)")
    MDVP_Flo = st.text_input("MDVP:Flo(Hz)")
    JitterPer = st.text_input("JitterPer")
    JitterAbs = st.text_input("JitterAbs")
    MDVP_RAP = st.text_input("MDVP_RAP")
    MDVP_PPQ = st.text_input("MDVP_PPQ")
    MDVP_DDP = st.text_input("MDVP_DDP")
    MDVP_Shimmer = st.text_input("MDVP_Shimmer")
    MDVP_Shimmerdb = st.text_input("MDVP_Shimmerdb")
    Shimmer_APQ3 = st.text_input("Shimmer_APQ3")
    Shimmer_APQ5 = st.text_input("Shimmer_APQ5")
    MDVP_APQ = st.text_input("MDVP_APQ")
    Shimmer_DDA = st.text_input("Shimmer_DDA")
    NHR = st.text_input("NHR")
    HNR = st.text_input("HNR")
    RPDE = st.text_input("RPDE")
    DFA = st.text_input("DFA")
    spread1 = st.text_input("spread1")
    spread2 = st.text_input("spread2")
    D2 = st.text_input("D2")
    PPE = st.text_input("PPE")

    ok = st.button("Submit")

    if ok:
        scaler = MinMaxScaler((-1, 1))
        X = np.array([[197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498, 0.01098, 0.09700, 0.00563,
                     0.00680, 0.00802, 0.01689, 0.00339, 26.77500, 0.422229, 0.741367, -7.348300, 0.177551, 1.743867, 0.085569]])
        X[:, 0] = le_MDVP_Fo.fit_transform(X[:, 0])
        X[:, 1] = le_MDVP_Fhi.fit_transform(X[:, 1])
        X[:, 2] = le_MDVP_Flo.fit_transform(X[:, 2])
        X[:, 3] = le_MDVP_JitterPer.fit_transform(X[:, 3])
        X[:, 4] = le_MDVP_JitterAbs.fit_transform(X[:, 4])
        X[:, 5] = le_MDVP_RAP.fit_transform(X[:, 5])
        X[:, 6] = le_MDVP_PPQ.fit_transform(X[:, 6])
        X[:, 7] = le_Jitter_DDP.fit_transform(X[:, 7])
        X[:, 8] = le_MDVP_Shimmer.fit_transform(X[:, 8])
        X[:, 9] = le_MDVP_Shimmer_dB.fit_transform(X[:, 9])
        X[:, 10] = le_Shimmer_APQ3.fit_transform(X[:, 10])
        X[:, 11] = le_Shimmer_APQ5.fit_transform(X[:, 11])
        X[:, 12] = le_MDVP_APQ.fit_transform(X[:, 12])
        X[:, 13] = le_Shimmer_DDA.fit_transform(X[:, 13])
        X[:, 14] = le_NHR.fit_transform(X[:, 14])
        X[:, 15] = le_HNR.fit_transform(X[:, 15])
        X[:, 16] = le_RPDE.fit_transform(X[:, 16])
        X[:, 17] = le_DFA.fit_transform(X[:, 17])
        X[:, 18] = le_spread1.fit_transform(X[:, 18])
        X[:, 19] = le_spread2.fit_transform(X[:, 19])
        X[:, 20] = le_D2.fit_transform(X[:, 20])
        X[:, 21] = le_PPE.fit_transform(X[:, 21])
        # input_data_as_numpy_array = np.asarray(input_data)
        # input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        # std_data = scaler.fit_transform(input_data_reshaped)
        prediction = model_xg.predict(X)
        print(prediction)
        print(prediction[0])
        if (prediction[0] == 1):
            st.subheader("The person has parkinson disease")
        else:
            st.subheader(
                "The person do not have parkinson disease ")
