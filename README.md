**word2vec.py** : 將資料從mongoDB取出，並做word to vector，會產生5個檔<br>
1. citedRecord_withID.txt<br>
    shcolarID + 英文文章數 + 記錄數 + 更新時間0 + 引用次數0 + 更新時間1 + 引用次數1 + ...<br>
2. data_withID.txt :<br>
    ID + 英文文章數 + 文字<br>
3. vector_withID.txt :<br>
    ID + vectors<br>
4. w2v_model.model<br>
5. w2v_vectorTable.txt<br>

**generate_data_to_txt.py** : 產生每個學者的"data"，輸入1或2決定要產生vector還是word的訓練資料<br>
假設學者有5筆紀錄，只會產生第1, 2, 3, 4筆紀錄的資料，因為第五筆沒有label<br>
1. dataRecord_vector_n.txt :<br>
    n+1次是否真的有更新(label) + ID + 更新時間n + 引用次數n + 第n次是否有更新(第1筆預設為1) + vectors<br>
2. dataRecord_word_n.txt : <br>
    n+1次是否真的有更新(label) + ID + 更新時間n + 引用次數n + 第n次是否有更新(第1筆預設為1) + words<br>

**generate_record_file.py** : 產生每個學者的紀錄檔<br>
1. citedRecord_withID.txt : <br>
    ID + 英文論文數 + 有幾筆紀錄 +[ 第 1 筆更新時間 + 第 1 筆的引用次數 ] + [ 第 2 筆更新時間 + 第 2 筆的引用次數 ] + ... + [ 第 n 筆更新時間 + 第 n 筆的引用次數 ]<br>