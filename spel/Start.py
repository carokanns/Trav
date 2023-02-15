import streamlit as st

st.set_page_config(
    page_title="Start",
    page_icon="👋",
)
st.sidebar.header("Välj från menyn ovan")
st.markdown("# Start av v75-applikation")
st.image('winning_horse.png')  # ,use_column_width=True)

st.write(
    """Enjoy!"""
    )

if st.button("Töm loggfilen"):
    with open("v75.log", "w") as f:
        f.write("")
