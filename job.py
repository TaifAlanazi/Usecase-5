#Import all relevant librarie
import streamlit as st
from PIL import  Image
import matplotlib as plt
import pandas as pd


df = pd.read_csv('data.csv')




st.title('Fresh Graduates Job Hunting: Understanding the Job Market in Saudi Arabia')
st.image('job.png')
st.markdown("""
### Looking for a job as a fresh graduate? Get a full understanding of Saudi Arabia's job market.
Entering adulthood after graduation and beginning your job search can feel overwhelming, especially when you're not familiar with the job market. 
Questions like *"Which cities have more opportunities?"* and *"What salary should I expect as a fresh graduate?"* may come to mind.
Don't worry, by the end of this, you'll have some clarity and answers to guide your job search.
""")


st.markdown("""
### First, let's find out which cities have the most job postings.
Understanding which cities have more job opportunities is an important first step in your job search, 
            to help you know where to start hunting, take a look at chart below and see which cities have more job postings than other 
""")

img = Image.open('job_postings_by_region.png')
st.image(img)


st.markdown("""
As you can see, Riyadh should be at the top of your list, with **42%** of the total job postings coming from companies there. 
Next is Makkah with **25%**, followed by the Eastern Region with **15%**. Consider these statistics when planning your job search.
""")


st.markdown("""
### Does your experience impact job opportunities?
You might be wondering, *"If I don’t have any experience, will I still be able to get a job?"*
Let's explore this question with the chart below.
""")

img2 = Image.open('job_postings_by_experience.png')
st.image(img2)

st.markdown("""
As shown above, job postings for fresh graduates (with zero experience) are still significant. 
There's a growing focus on creating opportunities for graduates, aligned with Saudi Arabia's Vision 2030 goals. 
So relax and know that there are plenty of opportunities available for you as a fresh graduate.
""")

st.markdown("""
### What salary should you expect as a fresh graduate?
We all aim for high salaries, but before deciding on your minimum salary, take a look at the chart below for some context.
""")

img3 = Image.open('histogram_salary_fresh_graduates.png')
st.image(img3)

st.markdown("""
As you can see, the salary for fresh graduates typically ranges between **4000 SAR** to **6000 SAR**. 
While it might feel a bit disappointing at first, remember this is just the beginning of your career, and your salary will grow as you gain more experience.
""")

st.markdown("""
### Does gender affect job opportunities?
There's often debate and controversy around whether being male or female impacts job opportunities. 
Let’s end this debate with one simple chart.
""")

img4 = Image.open('job_postings_by_gender.png')
st.image(img4)


st.markdown("""
As you can see from the chart, there's not much difference between the number of job postings for males and females. 
Let’s put this debate to rest, gender isn't as big of a factor as some may believe.
""")


st.markdown("""
### Final advice...
""")
st.image('ambition.png')
st.markdown("""
Finding a job may not always be easy, but with a clear set of short-term and long-term goals, 
an understanding of the job market, and a well-planned approach, you’ll find the process more manageable.
Stay patient, and keep pushing toward your goals. You can do it as everyone did before you.
Good luck!
""")