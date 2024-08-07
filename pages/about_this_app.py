import streamlit as st

st.markdown('Copyright (c) Jacopo Ventura, 2024. Distribution not allowed.', unsafe_allow_html=True)

st.markdown('<h2>About this app</h2>', unsafe_allow_html=True)

st.markdown('This application analyzes the price movement of assets in a specific time window. The output is a detailed report of the probabilities of the price movement up to numerous price change levels.', unsafe_allow_html=True)

st.markdown('When trading with options, the greek *delta* provides an empirical estimate of the trade success probaiblity. However, unexpected volatility can occur just right after the trade open, with the underlying price going in the opposite trade direction.
When this happens, it's crucial for the trader to stay calm and handle the trade correctly.
This is possible by knowing the probability of the price change of the underlying asset. With this data, better strike levels can be chosen.', unsafe_allow_html=True)
