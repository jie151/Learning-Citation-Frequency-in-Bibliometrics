**word2vec.py** : 將資料從 mongoDB 取出，並做 word to vector，會產生 4 個檔<br>

1. data_withID.txt :<br>
   ID + 英文文章數 + 文字<br>
2. vector_withID.txt :<br>
   ID + vectors<br>
3. w2v_model.model<br>
4. w2v_vectorTable.txt<br>

**generate_data_to_txt_add.py** : 產生每個學者的"data"，輸入 1 或 2 決定要產生 vector 還是 word 的訓練資料(包含 h_index, i10_index)<br>
假設學者有 5 筆紀錄，只會產生第 1, 2, 3, 4 筆紀錄的資料，因為第五筆沒有 label<br>
可以順便產生訓練\測試集資料<br>
1. dataRecord_vector_add.txt: <br>
   n+1 次是否真的有更新(label) + ID + 更新時間 n + 引用次數 n + h_index + i10_index + 第 n 次是否有更新(第 1 筆預設為 1) + vectors<br>

**generate_record_file_add.py** : 產生每個學者的紀錄檔，包含 h_index, i10_index<br>
1. citedRecord_withID_add.txt<br>
   ID + 英文論文數 + 有幾筆紀錄 +[ 第 1 筆更新時間 + 引用次數 + h_index, i10_index ] + [ 第 2 筆更新時間 + 引用次數 + h_index, i10_index ] + ... + [ 第 n 筆更新時間 + 引用次數 + h_index, i10_index ]<br>


