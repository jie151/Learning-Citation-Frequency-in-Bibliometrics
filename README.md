**word2vec.py** : 將資料從mongoDB取出，並做word to vector，會產生6個檔<br>
1. citedRecord_withID.txt<br>
    shcolarID + 英文文章數 + 記錄數 + 更新時間0 + 引用次數0 + 更新時間1 + 引用次數1 + ...<br>
2.  data.txt :<br>
    文字 (所有學者的文字，每位學者以"/n"分開)<br>
3. data_withID.txt :<br>
    ID + 英文文章數 + 文字<br>
4. vector_withID.txt :<br>
    ID + vectors<br>
5. w2v_model.model<br>
6. w2v_vectorTable.txt<br>

**generate_data_next_to_txt.py** : 得到一筆"data"，輸入1或2決定要產生vector還是word的訓練資料、輸入n決定要哪一筆紀錄，會產生一個檔(n從1開始)<br>
1. dataRecord_vector_n.txt :<br>
    n+1次是否真的有更新(label) + ID + 更新時間n + 引用次數n + 是否有更新(第0筆預設為1) + vectors<br>
2. dataRecord_word_n.txt : <br>
    n+1次是否真的有更新(label) + ID + 更新時間n + 引用次數n + 是否有更新(第0筆預設為1) + words<br>
