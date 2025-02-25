import streamlit as st
from parameters import PAGES
import os
import shutil

pages = PAGES

pg = st.navigation(pages)
pg.run()