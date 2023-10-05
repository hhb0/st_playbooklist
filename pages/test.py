from streamlit_elements import elements, mui, html
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from annotated_text import annotated_text
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.row import row
import pandas as pd
import numpy as np
import openai
import pinecone
import pickle
from pages.generate_result_img import *

st.markdown(
    """
    <style>
        .stSpinner > div {
            text-align:center;
            align-items: center;
            justify-content: center;
        }
    </style>""",
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=None)
def init_openai_key():
    openai.api_key = st.secrets.OPENAI_TOKEN

    return openai.api_key

with open('index_list.pickle', 'rb') as file:
    index_list = pickle.load(file)

def init_pinecone_connection():
    pinecone.init(
        api_key=st.secrets["PINECONE_KEY"],
        environment=st.secrets["PINECONE_REGION"]
    )
    pinecone_index = pinecone.Index('bookstore')
    return pinecone_index

pinecone_index = init_pinecone_connection()

@st.cache_data(show_spinner=None)
def generate_songs():
    df = pd.read_csv('./pages/data/melon_kakao_streamlit.csv')
    songs = df['song_name'] + ' | ' + df['artist_name_basket']

    return songs, df

def get_embedding(query):
    response = openai.Embedding.create(
        input=[query],
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def get_vectors_by_ids(pinecone_index, index_list):
    vector_data_list = []  # 벡터 데이터를 모을 리스트

    for s_id in index_list:
        # ID에 해당하는 벡터 데이터를 불러옴
        fetch_response = pinecone_index.fetch(ids=[str(s_id)], namespace="playbooklist")

        # 결과에서 벡터 데이터 추출
        if fetch_response["vectors"]:
            vector_data = fetch_response["vectors"][str(s_id)]["values"]
            vector_data_list.append(vector_data)

    return vector_data_list

def _vector_search(query_embedding):
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=20,
        include_metadata=True,
    )
    matches = results["matches"]
    return sorted([x["metadata"] for x in matches if x['metadata']['rating'] >= 8],
                  key=lambda x: (x['review_cnt'], x['rating']), reverse=True)[:5]

def generate_result():
    vector_data_list = get_vectors_by_ids(pinecone_index, index_list)
    index = [i for i in range(len(vector_data_list))]
    embedding_len = len((vector_data_list[0]))
    embeddings = np.array([0.0 for x in range(embedding_len)])
    for embedding in vector_data_list:
        embeddings += embedding
    result = _vector_search(list(embeddings))
    return result

cur_img_index = 0  # cur_img_index를 전역 변수로 초기화
img_paths = []  # img_paths를 전역 변수로 초기화

def show_image():
    global cur_img_index, img_paths
    if not img_paths:  # 이미지 경로가 없을 때만 초기화
        cur_img_index = 0
        img_paths = []

        result = generate_result()
        mockup_img = generate_mockup_img()
        for index in range(len(result)):
            img_url = result[index]['img_url']
            title = result[index]['title']
            authors = result[index]['authors']
            # 결과 이미지를 result_0.png, result_1.png로 저장. 덮어쓰기해서 용량 아끼기 위함.
            generate_result_img(index, mockup_img, img_url, title, authors)

        if result:
            for i in range(len(result)):
                img_paths.append(f"./pages/result_img/result_{i}.png")

    return cur_img_index, img_paths

cur_img_index, img_paths = show_image()

def get_author_title(item):
    return f"**{item['authors']}** | **{item['publisher']}**"

if __name__ == '__test__':
    openai.api_key = init_openai_key()

with st.spinner(text="**책장에서 책을 꺼내오고 있습니다..📚**"):
    with stylable_container(
            key="result_container",
            css_styles="""
            {
                border: 3px solid rgba(150, 55, 23, 0.2);
                border-radius: 0.5rem;
                padding: calc(1em - 1px)
            }
            """,
    ):
        c1, c2 = st.columns(2)
        result = generate_result()
        mockup_img = generate_mockup_img()
        index_list = [x for x in range(len(result))]
        i = 0
        with c1:
            for index in range(len(result)):
                img_url = result[index]['img_url']
                title = result[index]['title']
                authors = result[index]['authors']
                # 결과 이미지를 result_0.png, result_1.png로 저장. 덮어쓰기해서 용량 아끼기 위함.
                generate_result_img(index, mockup_img, img_url, title, authors)

            st.image(img_paths[index_list[i]])

            c3, c4 = st.columns(2)
            with c3:
                previous_img = st.button("**◀◀ 이전 장으로**")
            with c4:
                next_img = st.button("**다음 장으로 ▶▶**")

        with c2:
            want_to_main = st.button("새 플레이리스트 만들기 🔁")
            if want_to_main:
                switch_page("main")
            annotated_text(("**추천결과**", "", "#ff873d"))

            item = result[index_list[i]]
            st.header(item["title"])
            st.write(
                f"**{item['authors']}** | {item['publisher']} | {item['published_at']} | [yes24]({item['url']})")
            st.write(item["summary"])

        if previous_img:
            i -= 1
            if i < 0:
                i = len(img_paths) - 1

        if next_img:
            i += 1
            if i >= len(img_paths):
                i = 0

