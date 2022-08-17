*word2vec.py : 將資料從mongoDB取出，並做word to vector，會產生6個檔
    1. citedRecord_withID.txt
        shcolarID + 英文文章數 + 記錄數 + 更新時間0 + 引用次數0 + 更新時間1 + 引用次數1 + ...
    2.  data.txt :
        文字 (所有學者的文字，每位學者以"/n"分開)
    3. data_withID.txt :
        ID + 英文文章數 + 文字
    4. vector_withID.txt :
        ID + vectors
    5. w2v_model.model
    6. w2v_vectorTable.txt

*generate_data_to_txt.py : 得到一筆"data"，輸入n決定要哪一筆紀錄，會產生一個檔
    1. dataRecord_n.txt :
        更新時間n + 引用次數n + 是否有更新(第0筆預設為1)