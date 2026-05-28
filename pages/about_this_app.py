import streamlit as st
st.markdown('<h2 style="text-align: center;">About this app</h2>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify">This application analyzes the price movement of assets in a specific time window. '
            'The output is a detailed report of the probabilities of the price movement up to numerous price change levels.</div>',
            unsafe_allow_html=True)
st.markdown('<div> </div> ', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center;">Problem</h3>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify">I want to analyze the price movement of liquid assets to optimize the buy-the-dip method, '
            'by analyzing how deep corrections are in the stock market.</div>', unsafe_allow_html=True)
st.markdown('<div> </div> ', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center;">Solution</h3>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify">Given an input ticker and a selected time window, '
            'the app analyses the price movements for different durations (daily, weekly, '
            'monthly) and calculates the probability of the price change up to a certain price change level. '
            'The price movement is also analyzed for gap up / down scenarios and as function of the VIX level as well.</div>',
            unsafe_allow_html=True)
st.markdown('<div> </div> ', unsafe_allow_html=True)
st.markdown('<div> </div> ', unsafe_allow_html=True)
st.markdown('<div> </div> ', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center;">Copyright</h2>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center">Copyright (c) Jacopo Ventura, 2024. Distribution not allowed.</div>', unsafe_allow_html=True)
