import streamlit as st

pg = st.navigation([st.Page("EDA.py"), st.Page("Prediction Model.py")])
pg.run()


# Footer
st.markdown("---")
st.markdown("Developed by Camilla Louise Jensen, Esben Graahede, Imran Talukder, Piyal Dey, & Samil Demiroglu")