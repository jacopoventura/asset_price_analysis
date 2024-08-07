import streamlit as st

st.markdown('Copyright (c) Jacopo Ventura, 2024. Distribution not allowed.', unsafe_allow_html=True)

st.markdown('<h2 style="text-align: center;">About this app</h2>', unsafe_allow_html=True)

st.markdown('<div style="text-align: justify">This application analyzes the price movement of assets in a specific time window. The output is a detailed report of the probabilities of the price movement up to numerous price change levels.</div>', unsafe_allow_html=True)

st.markdown('<h3 style="text-align: center;">Problem</h3>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify">When trading with options, the greek *delta* provides an empirical estimate of the trade success probaiblity.</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify">However, unexpected volatility can occur just right after the trade open, with the underlying price going in the opposite trade direction. When this happens, it is crucial for the trader to stay calm and handle the trade correctly.</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify">This is possible by knowing the probability of the price change of the underlying asset. With this data, better strike levels can be chosen.</div>', unsafe_allow_html=True)

st.markdown('<h3 style="text-align: center;">Solution</h3>', unsafe_allow_html=True)

st.markdown('<div style="text-align: justify">Given an input ticker and a selected time window, the app analyses the price movements for different durations (daily, weekly, monthly) and calculates the probability of the price change up to a certain price change level. The price movement is also analyzed for gap up / down scenarios and as function of the VIX level as well.</div>', unsafe_allow_html=True)

st.markdown('<h3 style="text-align: center;">Example of usage</h3>', unsafe_allow_html=True)
