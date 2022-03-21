import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import streamlit as st
import streamlit.components.v1 as stc



def main():
    current_path=os.getcwd()
    st.write(current_path)
    
if __name__ == '__main__':
	main()

    



