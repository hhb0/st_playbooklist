import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from annotated_text import annotated_text
from google.cloud import translate
from google.oauth2.service_account import Credentials
import pinecone
import pandas as pd
import pickle
import openai
import time
import traceback

st.set_page_config(
    page_title="main",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
        .stProgress > div > div > div > div {
            background-color: orange;
        }
    </style>""",
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=None)
def init_openai_key():
    openai.api_key = st.secrets.OPENAI_TOKEN

    return openai.api_key

def init_gcp_connection():
    credentials = Credentials.from_service_account_info(st.secrets.GOOGLE)
    translate_client = translate.TranslationServiceClient(credentials=credentials)

    return translate_client

def init_pinecone_connection():
    pinecone.init(
        api_key=st.secrets["PINECONE_KEY"],
        environment=st.secrets["PINECONE_REGION"]
    )
    pinecone_index = pinecone.Index('bookstore')
    return pinecone_index


@st.cache_data(show_spinner=None)
def generate_songs():
    df = pd.read_csv('./pages/data/melon_kakao_streamlit.csv')
    songs = df['song_name'] + ' | ' + df['artist_name_basket']

    return songs, df


def generate_index_list(songs, df):
    st.title('곡 선택')
    selected_list = st.multiselect('원하는 곡을 선택하세요 ( 최대 5곡 ):', songs, key="multiselect_songs")
    index_list = []

    for row in selected_list:
        row_index = list(songs).index(row)
        index_id = df.loc[row_index, 'song_id']
        index_list.append(str(index_id))

    return index_list

@st.cache_data(show_spinner=None)
def display_song_information(index_id):
    album_id = df[df["song_id"] == int(index_id)]["album_id"].values[0]
    st.image(f"./pages/album_img/{album_id}.png")
    annotated_text(("Song Name", "", "#ff873d"))
    st.write(df[df["song_id"] == int(index_id)]["song_name"].values[0])

def get_translation(query):
    parent = f"projects/{st.secrets.PROJECTID}/locations/global"
    response = translate_client.translate_text(
        request={
            "parent": parent,
            "contents": [query],
            "mime_type": "text/plain",
            "source_language_code": "ko",
            "target_language_code": "en-US",
        }
    )
    translations = response.translations
    return translations[0].translated_text


def vector_search(query_embedding):
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=20,
        include_metadata=True,
    )
    matches = results["matches"]
    return sorted([x["metadata"] for x in matches if x['metadata']['rating'] >= 8], # 메타 필터링
                  key=lambda x: (x['review_cnt'], x['rating']), reverse=True)[:3] # 정렬, 반대는 - 붙이기

def get_embedding(query):
    response = openai.Embedding.create(
        input=[query],
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def check_and_add_vector(pinecone_index, index_list, df):
    for s_id in index_list:
        fetch_response = pinecone_index.fetch(ids=[str(s_id)], namespace="playbooklist")
        if not fetch_response["vectors"]:
            s_contents = df[df["song_id"] == int(s_id)]["total_contents"].values[0]
            s_eng = get_translation(query=s_contents)
            s_embedding = get_embedding(query=s_eng)
            pinecone_index.upsert([(str(s_id), s_embedding)], namespace="playbooklist")

if __name__ == '__main__':
    openai.api_key = init_openai_key()
    translate_client = init_gcp_connection()
    pinecone_index = init_pinecone_connection()
    submit_button = False

    empty1, con1, empty2 = st.columns([0.3, 1.0, 0.3])
    with empty1:
        st.empty()
    with con1:
        want_to_contribute = st.button("🏠home")
        if want_to_contribute:
            switch_page("home")
    with empty2:
        st.empty()

    # if 'i' not in st.session_state:
    #     st.session_state['i'] = []

    empty1, con1, empty2 = st.columns([0.2, 1.0, 0.2])
    with empty1:
        st.empty()
    with con1:
        songs, df = generate_songs()
        index_list = generate_index_list(songs, df)
    with empty2:
        st.empty()

    co1, co2, co3 = st.columns([0.2, 1.0, 0.2])
    with co1:
        st.empty()
    with co2:
        container = st.empty()
        form = container.form("my_form", clear_on_submit=True)
        if index_list:

            with form:
                c7, c8, c9 = st.columns(3)
                with c7:
                    submit_button = st.form_submit_button("플레이북리스트 결과보기 >")
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    display_song_information(index_list[0])
                with c2:
                    try:
                        display_song_information(index_list[1])
                    except IndexError:
                        pass
                with c3:
                    try:
                        display_song_information(index_list[2])
                    except IndexError:
                        pass
                with c4:
                    try:
                        display_song_information(index_list[3])
                    except IndexError:
                        pass
                with c5:
                    try:
                        display_song_information(index_list[4])
                    except IndexError:
                        pass

    with co3:
        st.empty()

    with st.container():
        e1, c1, e2 = st.columns([0.2, 1.0, 0.2])
        with e1:
            st.empty()
        if index_list:
            with c1:
                if submit_button:
                    progress_text = "당신을 위한 책을 찾고 있습니다...🔍"
                    my_bar = st.progress(0, text=progress_text)

                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1, text=progress_text)
                    time.sleep(1)
                    with my_bar:
                        check_and_add_vector(pinecone_index, index_list, df)
                        my_bar.empty()
                        st.success("Play Book List🎼")
                        with open('index_list.pickle', 'wb') as file:
                            pickle.dump(index_list, file)
                        switch_page("result")
        with e2:
            st.empty()
